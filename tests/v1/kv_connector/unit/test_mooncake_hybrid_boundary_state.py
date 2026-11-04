# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Mooncake P/D for Mamba hybrids in align mode with ``ssm_boundary_state_xfer``:
P also sends the state at the prompt's last block boundary, so a later turn
sent to the decode pod alone gets a prefix hit. Drives real P and D schedulers.

Unlike NIXL, the router hands P and D their params independently, so both
sides derive the boundary from the prompt length: P computes and D loads the
first N-1 tokens, and the boundary is the last block multiple below N-1."""

from unittest.mock import patch

import pytest

from vllm.platforms import current_platform
from vllm.sampling_params import SamplingParams
from vllm.utils.hashing import sha256
from vllm.v1.attention.backends.utils import NULL_BLOCK_ID
from vllm.v1.core.kv_cache_utils import get_request_block_hasher
from vllm.v1.kv_cache_interface import KVCacheGroupSpec
from vllm.v1.outputs import ModelRunnerOutput
from vllm.v1.request import Request, RequestStatus

from .utils import (
    EOS_TOKEN_ID,
    create_model_runner_output,
    create_request,
    create_scheduler,
    create_vllm_config,
    make_kv_cache_config,
)

pytestmark = pytest.mark.cpu_test

BLOCK_SIZE = 16
FA_GROUP, MAMBA_GROUP = 0, 1
P_ENGINE = "prefill-engine"


@pytest.fixture(autouse=True)
def _cpu_device_type():
    with patch.object(current_platform, "device_type", "cpu"):
        yield


def _make_scheduler(
    kv_role: str, num_mamba_groups: int = 1, ssm_boundary_state_xfer: bool = True
):
    vllm_config = create_vllm_config(
        block_size=BLOCK_SIZE,
        max_num_batched_tokens=256,
        kv_connector="MooncakeConnector",
        kv_role=kv_role,
        kv_connector_extra_config={"ssm_boundary_state_xfer": ssm_boundary_state_xfer},
    )
    vllm_config.scheduler_config.disable_hybrid_kv_cache_manager = False
    vllm_config.scheduler_config.async_scheduling = False
    vllm_config.cache_config.mamba_cache_mode = "align"
    kv_cache_config = make_kv_cache_config(
        block_size=BLOCK_SIZE,
        mamba_enabled=True,
        mamba_cache_mode="align",
        num_blocks=200,
    )
    kv_cache_config.hash_block_size = BLOCK_SIZE
    mamba_group = kv_cache_config.kv_cache_groups[MAMBA_GROUP]
    for i in range(1, num_mamba_groups):
        kv_cache_config.kv_cache_groups.append(
            KVCacheGroupSpec([f"mamba_extra{i}"], mamba_group.kv_cache_spec)
        )
    return create_scheduler(
        vllm_config, kv_cache_config=kv_cache_config, hash_block_size=BLOCK_SIZE
    )


def _make_request(request_id: str, prompt: list[int], max_tokens: int) -> Request:
    # Ensures the NONE hash is initialized, as create_request does.
    create_request(num_tokens=1, block_size=BLOCK_SIZE)
    sampling_params = SamplingParams(max_tokens=max_tokens)
    sampling_params.update_from_generation_config({}, EOS_TOKEN_ID)
    return Request(
        request_id=request_id,
        prompt_token_ids=list(prompt),
        sampling_params=sampling_params,
        pooling_params=None,
        mm_features=None,
        block_hasher=get_request_block_hasher(BLOCK_SIZE, sha256),
    )


def _real(block_ids: list[int]) -> list[int]:
    return [b for b in block_ids if b != NULL_BLOCK_ID]


def _step(scheduler, request: Request):
    """One scheduler step; samples a token once the prompt is computed."""
    scheduler_output = scheduler.schedule()
    if request.num_computed_tokens >= request.num_prompt_tokens:
        model_output = create_model_runner_output([request])
    else:
        model_output = ModelRunnerOutput(
            req_ids=[request.request_id],
            req_id_to_index={request.request_id: 0},
            sampled_token_ids=[[]],
        )
    return scheduler.update_from_output(scheduler_output, model_output)


def _run_prefill(prefiller, prompt: list[int], request_id: str = "p-1"):
    """Prefill ``prompt`` on P until it finishes; return the request and the
    block ids P will send for it, per transfer group."""
    request = _make_request(request_id, prompt, max_tokens=1)
    request.kv_transfer_params = dict(
        do_remote_prefill=False, do_remote_decode=True, transfer_id=f"x-{request_id}"
    )
    prefiller.add_request(request)
    for _ in range(8):
        _step(prefiller, request)
        if request.is_finished():
            break
    assert request.is_finished(), "prefill did not finish"
    meta = prefiller.schedule().kv_connector_metadata
    _, send_block_ids = meta.reqs_to_send[request_id]
    return request, send_block_ids


def _run_decode(
    decoder, prompt: list[int], transfer_id: str, num_output: int, request_id="d-1"
):
    """Receive ``prompt`` from P, decode ``num_output`` tokens and finish.
    Returns the local SSM block ids D receives into and the generated tokens."""
    request = _make_request(request_id, prompt, max_tokens=num_output)
    request.kv_transfer_params = dict(
        do_remote_prefill=True,
        do_remote_decode=False,
        remote_engine_id=P_ENGINE,
        remote_bootstrap_addr="http://prefill:8998",
        transfer_id=transfer_id,
    )
    decoder.add_request(request)

    scheduler_output = decoder.schedule()
    recv = scheduler_output.kv_connector_metadata.reqs_to_recv[P_ENGINE][request_id]
    local_ssm_block_ids = list(recv.local_block_ids[MAMBA_GROUP])
    decoder.update_from_output(
        scheduler_output,
        create_model_runner_output([], finished_recving={request_id}),
    )

    output_tokens: list[int] = []
    for i in range(num_output):
        scheduler_output = decoder.schedule()
        token = 1000 + i
        last = i == num_output - 1
        decoder.update_from_output(
            scheduler_output,
            create_model_runner_output([request], use_eos=last, token_id=token),
        )
        output_tokens.append(EOS_TOKEN_ID if last else token)
    decoder.schedule()
    assert request.is_finished()
    return local_ssm_block_ids, output_tokens


def _decode_pod_hit(decoder, prompt: list[int], output_tokens: list[int]) -> int:
    turn2 = prompt + output_tokens + [500 + i for i in range(10)]
    request = _make_request("d-next", turn2, max_tokens=1)
    _, num_hit_tokens, _ = decoder.kv_cache_manager.get_computed_blocks(request)
    return num_hit_tokens


def _turn1_prompt(num_tokens: int) -> list[int]:
    return [7 + i for i in range(num_tokens)]


def test_decode_pod_hits_prompt_boundary_after_mooncake_receive():
    """58-token prompt, block 16: P computes 57 tokens and also sends its
    state at 48, so the decode pod resumes the next turn from 48."""
    prefiller = _make_scheduler("kv_producer")
    decoder = _make_scheduler("kv_consumer")
    prompt = _turn1_prompt(58)

    _, sent = _run_prefill(prefiller, prompt)
    assert len(_real(sent[MAMBA_GROUP])) == 2

    received, output_tokens = _run_decode(decoder, prompt, "x-p-1", num_output=3)
    assert len(_real(received)) == 2
    assert _decode_pod_hit(decoder, prompt, output_tokens) == 48


@pytest.mark.parametrize(
    "p_xfer,d_xfer", [(False, False), (True, False), (False, True)]
)
def test_boundary_state_needs_flag_on_both_sides(p_xfer, d_xfer):
    """Off by default. P and D cannot tell each other whether the flag is on:
    P's extra block is dropped by the worker (it pairs P's trailing blocks),
    while D's extra slot makes P report "P num blocks less than D" and the pull
    fail. Either way the decode pod caches no boundary state."""
    prefiller = _make_scheduler("kv_producer", ssm_boundary_state_xfer=p_xfer)
    decoder = _make_scheduler("kv_consumer", ssm_boundary_state_xfer=d_xfer)
    prompt = _turn1_prompt(58)

    _, sent = _run_prefill(prefiller, prompt)
    received, output_tokens = _run_decode(decoder, prompt, "x-p-1", num_output=2)
    assert len(_real(sent[MAMBA_GROUP])) == 1 + p_xfer
    assert len(_real(received)) == 1 + d_xfer
    if not d_xfer:
        assert _decode_pod_hit(decoder, prompt, output_tokens) == 0


def test_prompt_ending_on_boundary_sends_running_state_only():
    """49 tokens: P computes exactly 48, so its running state is the boundary
    state and one slot is enough."""
    prefiller = _make_scheduler("kv_producer")
    decoder = _make_scheduler("kv_consumer")
    prompt = _turn1_prompt(49)

    _, sent = _run_prefill(prefiller, prompt)
    assert len(_real(sent[MAMBA_GROUP])) == 1

    received, output_tokens = _run_decode(decoder, prompt, "x-p-1", num_output=2)
    assert len(_real(received)) == 1
    assert _decode_pod_hit(decoder, prompt, output_tokens) == 48


def test_prefiller_holds_boundary_state_until_sent():
    """The boundary block must not be recycled while D may still read it."""
    prefiller = _make_scheduler("kv_producer")
    request, sent = _run_prefill(prefiller, _turn1_prompt(58))
    boundary = prefiller.kv_cache_manager.block_pool.blocks[sent[MAMBA_GROUP][0]]
    assert boundary.ref_cnt > 0

    scheduler_output = prefiller.schedule()
    prefiller.update_from_output(
        scheduler_output,
        create_model_runner_output([], finished_sending={request.request_id}),
    )
    prefiller.schedule()
    assert boundary.ref_cnt == 0
    assert boundary.block_hash is not None


def test_aborted_prefill_releases_boundary_pin():
    prefiller = _make_scheduler("kv_producer")
    pool = prefiller.kv_cache_manager.block_pool
    num_free = pool.get_num_free_blocks()
    request = _make_request("p-1", _turn1_prompt(58), max_tokens=1)
    request.kv_transfer_params = dict(
        do_remote_prefill=False, do_remote_decode=True, transfer_id="x-p-1"
    )
    prefiller.add_request(request)
    _step(prefiller, request)
    prefiller.finish_requests(request.request_id, RequestStatus.FINISHED_ABORTED)
    assert pool.get_num_free_blocks() == num_free
    assert not prefiller.get_kv_connector().connector_scheduler._boundary_pins


def test_prefiller_prefix_hit_on_boundary_still_sends_it():
    """P already caches the prompt, so its hit ends on the boundary and no new
    boundary state is computed or offered. P must send the hit block, or a
    cold decode pod's pull fails."""
    prefiller = _make_scheduler("kv_producer")
    prompt = _turn1_prompt(58)
    warm = _make_request("warm", prompt, max_tokens=1)
    prefiller.add_request(warm)
    for _ in range(8):
        _step(prefiller, warm)
        if warm.is_finished():
            break

    # P computes the first 57 tokens; its cache already ends on the boundary.
    probe = _make_request("probe", prompt[:57], max_tokens=1)
    assert prefiller.kv_cache_manager.get_computed_blocks(probe)[1] == 48

    _, sent = _run_prefill(prefiller, prompt, request_id="p-2")
    assert len(_real(sent[MAMBA_GROUP])) == 2

    decoder = _make_scheduler("kv_consumer")
    received, output_tokens = _run_decode(decoder, prompt, "x-p-2", num_output=2)
    assert len(_real(received)) == 2
    assert _decode_pod_hit(decoder, prompt, output_tokens) == 48


def test_decoder_local_hit_past_boundary_reserves_nothing():
    """A decode pod that already caches the prompt has the boundary state."""
    prefiller = _make_scheduler("kv_producer")
    decoder = _make_scheduler("kv_consumer")
    prompt = _turn1_prompt(58)

    _run_prefill(prefiller, prompt)
    _run_decode(decoder, prompt, "x-p-1", num_output=2)

    _run_prefill(prefiller, prompt, request_id="p-2")
    received, _ = _run_decode(decoder, prompt, "x-p-2", num_output=2, request_id="d-2")
    assert len(_real(received)) == 1


def test_every_mamba_group_sends_and_caches_its_boundary_state():
    """Hybrids such as Qwen3.5 have several GDN groups; the hit needs each."""
    prefiller = _make_scheduler("kv_producer", num_mamba_groups=3)
    decoder = _make_scheduler("kv_consumer", num_mamba_groups=3)
    prompt = _turn1_prompt(58)

    _, sent = _run_prefill(prefiller, prompt)
    assert [len(_real(g)) for g in sent[1:]] == [2, 2, 2]

    _, output_tokens = _run_decode(decoder, prompt, "x-p-1", num_output=3)
    assert _decode_pod_hit(decoder, prompt, output_tokens) == 48
