# Copyright (c) 2026 Intel Corporation. All Rights Reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for the XPU timing helper used by the patched FBGEMM benchmarks.

The failure this guards against is the one upstream's helper has on XPU: a
wall clock without device synchronisation measures queue submission, and
reports a figure that is wrong but plausible. A copy of known size gives a
physical lower bound on the time - it cannot finish faster than any XPU's
memory bandwidth allows - and submission alone comes in far under it.
"""

import logging
import time

import pytest
import torch
from fbgemm_xpu.bench.bench_utils import benchmark_torch_function

pytestmark = pytest.mark.skipif(
    not torch.xpu.is_available(), reason="requires an XPU device"
)

# Well above the HBM bandwidth of any current Intel GPU (PVC is ~3.3 TB/s).
_MAX_PLAUSIBLE_BYTES_PER_S = 10e12


@pytest.fixture(autouse=True)
def _release_device_memory():
    # The caching allocator keeps the copy operands below reserved after a test
    # ends. Return them to the driver: the GPU may be shared with another test
    # process, and later tests size their operands to the device's free memory.
    yield
    torch.xpu.empty_cache()


def test_time_is_bounded_by_device_bandwidth():
    # 1 GiB moved puts the bound near 100 us, an order of magnitude above the
    # submission time a wall clock without synchronisation would report, while
    # keeping the footprint modest on a 12 GB part shared with another process.
    src = torch.empty(128 * 1024 * 1024, dtype=torch.float, device="xpu")
    dst = torch.empty_like(src)
    seconds, _ = benchmark_torch_function(dst.copy_, (src,), iters=5)

    moved = 2 * src.numel() * src.element_size()
    assert seconds > moved / _MAX_PLAUSIBLE_BYTES_PER_S  # nosec B101


def _copy_seconds(numel, **kwargs):
    # Random, not uninitialised: freshly allocated memory is typically zeroed,
    # and memory compression can then report more than the real bandwidth.
    src = torch.rand(numel, dtype=torch.float, device="xpu")
    dst = torch.empty_like(src)
    seconds, _ = benchmark_torch_function(dst.copy_, (src,), iters=20, **kwargs)
    return seconds


def test_default_flush_is_twice_last_level_cache(monkeypatch):
    # The default flush must be sized from the device and run ahead of every
    # timed iteration. Upstream's fixed 40 MB leaves operands resident on PVC
    # (192 MB L2). This is checked on the helper's own calls rather than on
    # measured bandwidth, so it holds on any part and on a GPU shared with
    # another process, where timings carry no information about the cache.
    llc_bytes = torch.xpu.get_device_properties().last_level_cache_size
    flush_bytes = (2 * llc_bytes >> 20) << 20
    events = []

    real_neg_ = torch.Tensor.neg_
    real_sum = torch.sum

    def spy_neg_(self):
        if self.numel() * self.element_size() == flush_bytes:
            events.append("write")
        return real_neg_(self)

    def spy_sum(input, *args, **kwargs):
        if input.numel() * input.element_size() == flush_bytes:
            events.append("read")
        return real_sum(input, *args, **kwargs)

    monkeypatch.setattr(torch.Tensor, "neg_", spy_neg_)
    monkeypatch.setattr(torch, "sum", spy_sum)

    x = torch.zeros(1024, device="xpu")

    def f():
        events.append("f")
        return x.add_(1)

    iters, num_warmups = 5, 2
    benchmark_torch_function(f, (), iters=iters, num_warmups=num_warmups)

    assert "write" in events  # nosec B101
    first_flush = events.index("write")
    warmups, timed = events[:first_flush], events[first_flush:]
    assert warmups == ["f"] * num_warmups  # nosec B101
    # The timed loop may be repeated with a longer GPU lead, so the count of
    # timed calls is a multiple of iters. Each pass writes one buffer and then
    # reads the other, and each timed call follows a read, so the cache holds
    # clean lines when it starts.
    assert timed.count("f") % iters == 0  # nosec B101
    for previous, event in zip(timed, timed[1:]):
        if event == "f":
            assert previous == "read"  # nosec B101
        elif event == "read":
            assert previous == "write"  # nosec B101


def test_default_flush_evicts_last_level_cache():
    # The same copy, sized to half the last-level cache, timed with the
    # default flush, with no flush, and with a reference flush four times the
    # default, which leaves nothing of the copy in cache on any part. The
    # default must cover at least half of the gap between the other two.
    # Upstream's fixed 40 MB flush fails this on PVC (192 MB L2): the operands
    # stay resident and the flushed copy is as fast as the cached one.
    #
    # The reference, rather than a fixed ratio to the cached copy, sets what
    # eviction costs: that varies by part, as the flush leaves the cache clean
    # and a short copy's writes are absorbed by the write-back cache. A PVC
    # tile takes 2.4x as long for the evicted copy, a B60 only ~1.14x.
    #
    # The comparison only means something if the cached copy takes markedly
    # longer than the helper reports for any kernel at all. On a GPU shared
    # with another process, time-slicing inflates every measurement by the
    # same amount and both copies collapse onto that floor; so does a copy too
    # small for the part's cache to resolve. Neither says anything about the
    # flush, so the test is skipped rather than failed; the check above still
    # covers the flush's size and placement.
    llc_bytes = torch.xpu.get_device_properties().last_level_cache_size
    llc_floats = llc_bytes // 4
    floor_s = _copy_seconds(1, flush_gpu_cache_size_mb=0)
    cached_s = _copy_seconds(llc_floats // 4, flush_gpu_cache_size_mb=0)
    if cached_s < 1.5 * floor_s:
        pytest.skip(
            f"cached copy ({cached_s * 1e6:.0f} us) is at the launch floor "
            f"({floor_s * 1e6:.0f} us): the GPU is shared or the cache is too "
            "small to resolve, so the flush cannot be observed"
        )
    reference_s = _copy_seconds(
        llc_floats // 4, flush_gpu_cache_size_mb=8 * llc_bytes >> 20
    )
    if reference_s < 1.05 * cached_s:
        pytest.skip(
            f"evicted copy ({reference_s * 1e6:.0f} us) is within 5% of the "
            f"cached one ({cached_s * 1e6:.0f} us), so the flush cannot be "
            "observed"
        )
    flushed_s = _copy_seconds(llc_floats // 4)
    assert flushed_s - cached_s >= 0.5 * (reference_s - cached_s)  # nosec B101


def test_default_flush_adds_no_write_back_to_timed_call():
    # A copy four times the last-level cache starts from memory either way, so
    # the flush ahead of it must not make it slower. A flush that left the
    # cache full of dirty lines did: the copy wrote them back inside the timed
    # window, 8% slower than with no flush on a PVC tile and 6% on a B60. A
    # clean flush is 5-7% faster than none, as no flush leaves the previous
    # iteration's writes to be written back.
    llc_floats = torch.xpu.get_device_properties().last_level_cache_size // 4
    unflushed_s = _copy_seconds(4 * llc_floats, flush_gpu_cache_size_mb=0)
    flushed_s = _copy_seconds(4 * llc_floats)
    assert flushed_s <= 1.02 * unflushed_s  # nosec B101


def test_returns_output_and_passes_kwargs():
    x = torch.arange(8, dtype=torch.float, device="xpu")
    _, out = benchmark_torch_function(torch.add, (x, x), kwargs={"alpha": 2})
    torch.testing.assert_close(out.cpu(), (x + 2 * x).cpu())


def test_host_submission_latency_is_not_timed(caplog):
    # f spends 2 ms on the host before launching a tiny kernel. Without a GPU
    # lead ahead of the start event, the GPU idles through those 2 ms inside
    # the timed window; this is what a slow host does to small shapes on BMG.
    x = torch.zeros(1024, device="xpu")

    def host_heavy():
        time.sleep(2e-3)
        return x.add_(1)

    with caplog.at_level(logging.WARNING):
        seconds, _ = benchmark_torch_function(host_heavy, (), iters=5)
    assert seconds < 0.5e-3  # nosec B101
    assert "waited for the host" not in caplog.text  # nosec B101


def test_warns_when_f_synchronises(caplog):
    x = torch.zeros(1024, device="xpu")

    def synchronising():
        x.add_(1)
        torch.xpu.synchronize()

    with caplog.at_level(logging.WARNING):
        benchmark_torch_function(synchronising, (), iters=2, flush_gpu_cache_size_mb=0)
    assert "waited for the host" in caplog.text  # nosec B101


@pytest.mark.parametrize("device", ["cpu", "mtia"])
def test_rejects_non_xpu_device(device):
    with pytest.raises(ValueError, match="XPU only"):
        benchmark_torch_function(torch.zeros, (1,), device=device)
