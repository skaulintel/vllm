# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Host staging for Mooncake on accelerators its transports cannot address.

Mooncake's TCP transport reads and writes registered buffers with plain
``memcpy``/``send``. That is fine for CUDA builds (they special-case device
memory) but XPU device USM pointers are not host-accessible, so every put
fails with EFAULT and the local copy path segfaults. This wrapper mirrors each
registered KV region in pinned host memory, hands Mooncake the mirror, and
copies the touched byte ranges device<->host around every put/get. The layout
code keeps computing device addresses; they are translated here.
"""

import bisect
from collections import defaultdict
from typing import Any

import torch

from vllm.logger import init_logger

logger = init_logger(__name__)

# Mooncake's TCP transport queues at most MC_TCP_MAX_QUEUED_TRANSFERS_PER_PEER
# (default 1024) slices per peer and wedges, instead of failing, when one batch
# overflows it. Each key is one slice per layer, so split batches by slice count.
_MAX_SLICES_PER_CALL = 512


class _Region:
    def __init__(self, device_buf: torch.Tensor, host_buf: torch.Tensor):
        self.dev = device_buf
        self.host = host_buf
        self.dev_base = device_buf.data_ptr()
        self.host_base = host_buf.data_ptr()
        self.nbytes = device_buf.numel()


class HostStagedStore:
    """Proxy for ``MooncakeDistributedStore`` that stages KV through host RAM."""

    def __init__(self, store: Any):
        self._store = store
        self._regions: list[_Region] = []
        self._bases: list[int] = []

    def __getattr__(self, name: str) -> Any:
        return getattr(self._store, name)

    def register_device_storage(self, storage: torch.UntypedStorage) -> int:
        dev = torch.empty(0, dtype=torch.uint8, device=storage.device).set_(storage)
        host = torch.empty(dev.numel(), dtype=torch.uint8, pin_memory=True)
        ret = self._store.register_buffer(host.data_ptr(), host.numel())
        if ret != 0:
            return ret
        region = _Region(dev, host)
        idx = bisect.bisect(self._bases, region.dev_base)
        self._bases.insert(idx, region.dev_base)
        self._regions.insert(idx, region)
        logger.debug(
            "Mooncake host staging: device %#x -> host %#x (%d bytes)",
            region.dev_base,
            region.host_base,
            region.nbytes,
        )
        return 0

    def _region_for(self, addr: int, size: int) -> _Region:
        idx = bisect.bisect(self._bases, addr) - 1
        if idx >= 0:
            region = self._regions[idx]
            if addr + size <= region.dev_base + region.nbytes:
                return region
        raise ValueError(f"address {addr:#x}+{size} is not in a registered KV region")

    def _translate(
        self, addrs: list[list[int]], sizes: list[list[int]]
    ) -> tuple[list[list[int]], dict[_Region, list[tuple[int, int]]]]:
        host_addrs: list[list[int]] = []
        ranges: dict[_Region, list[tuple[int, int]]] = defaultdict(list)
        for key_addrs, key_sizes in zip(addrs, sizes, strict=True):
            out = []
            for addr, size in zip(key_addrs, key_sizes, strict=True):
                region = self._region_for(addr, size)
                offset = addr - region.dev_base
                out.append(region.host_base + offset)
                ranges[region].append((offset, size))
            host_addrs.append(out)
        return host_addrs, ranges

    @staticmethod
    def _coalesce(ranges: list[tuple[int, int]]) -> list[tuple[int, int]]:
        merged: list[tuple[int, int]] = []
        for offset, size in sorted(ranges):
            if merged and offset <= merged[-1][0] + merged[-1][1]:
                start, length = merged[-1]
                merged[-1] = (start, max(length, offset + size - start))
            else:
                merged.append((offset, size))
        return merged

    def _copy(self, ranges: dict[_Region, list[tuple[int, int]]], to_host: bool):
        devices = set()
        for region, spans in ranges.items():
            for offset, size in self._coalesce(spans):
                dev = region.dev[offset : offset + size]
                host = region.host[offset : offset + size]
                if to_host:
                    host.copy_(dev, non_blocking=True)
                else:
                    dev.copy_(host, non_blocking=True)
            devices.add(region.dev.device)
        for device in devices:
            torch.accelerator.current_stream(device).synchronize()

    @staticmethod
    def _split(addrs: list[list[int]]) -> list[slice]:
        batches, start, slices = [], 0, 0
        for i, key_addrs in enumerate(addrs):
            if slices and slices + len(key_addrs) > _MAX_SLICES_PER_CALL:
                batches.append(slice(start, i))
                start, slices = i, 0
            slices += len(key_addrs)
        batches.append(slice(start, len(addrs)))
        return batches

    def batch_put_from_multi_buffers(self, keys, addrs, sizes, *args, **kwargs):
        host_addrs, ranges = self._translate(addrs, sizes)
        self._copy(ranges, to_host=True)
        res: list[int] = []
        for b in self._split(host_addrs):
            res.extend(
                self._store.batch_put_from_multi_buffers(
                    keys[b], host_addrs[b], sizes[b], *args, **kwargs
                )
            )
        return res

    def batch_get_into_multi_buffers(self, keys, addrs, sizes, *args, **kwargs):
        host_addrs, _ = self._translate(addrs, sizes)
        res: list[int] = []
        for b in self._split(host_addrs):
            res.extend(
                self._store.batch_get_into_multi_buffers(
                    keys[b], host_addrs[b], sizes[b], *args, **kwargs
                )
            )
        # Only copy back keys that actually landed; failed ones are recomputed.
        ok = [i for i, r in enumerate(res) if r >= 0]
        if ok:
            _, ranges = self._translate([addrs[i] for i in ok], [sizes[i] for i in ok])
            self._copy(ranges, to_host=False)
        return res
