# Copyright (c) 2026 Intel Corporation. All Rights Reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""``all_to_one_device`` on XPU.

TorchRec's ``SeqEmbeddingsAllToOne`` gathers per-rank sequence embedding
outputs onto one device with this operator. FBGEMM's patched
``test_all_to_one_device`` checks the gathered values for inputs spread over
devices. These tests cover what it does not check: tensors already on the
target come back as the same views, copies are contiguous, invalid devices are
rejected, and copies work on non-default streams. The cross-device tests need
two XPU devices.
"""

import fbgemm_xpu  # noqa: F401  - registers the fbgemm XPU operators
import pytest
import torch

pytestmark = pytest.mark.skipif(
    not torch.xpu.is_available(), reason="requires an XPU device"
)

requires_two_xpus = pytest.mark.skipif(
    torch.xpu.is_available() and torch.xpu.device_count() < 2,
    reason="requires two XPU devices",
)


def test_same_device_preserves_views_and_empty_tensors():
    xpu = torch.device("xpu:0")
    torch.xpu.set_device(xpu)
    base = torch.arange(24, device=xpu).reshape(4, 6)
    tensors = [
        base,
        base[:, ::2],
        torch.empty((0, 5), device=xpu),
        torch.ones((), device=xpu),
    ]
    outputs = torch.ops.fbgemm.all_to_one_device(tensors, xpu)
    assert len(outputs) == len(tensors)  # nosec B101
    for actual, expected in zip(outputs, tensors):
        torch.testing.assert_close(actual, expected)
        assert actual.data_ptr() == expected.data_ptr()  # nosec B101
        assert actual.stride() == expected.stride()  # nosec B101


def test_rejects_unsupported_device_inputs():
    xpu = torch.device("xpu:0")
    tensor = torch.ones(2, device=xpu)
    with pytest.raises(RuntimeError, match="target_device must be XPU"):
        torch.ops.fbgemm.all_to_one_device([tensor], torch.device("cpu"))
    with pytest.raises(RuntimeError, match="Please pass target_device with device index"):
        torch.ops.fbgemm.all_to_one_device([tensor], torch.device("xpu"))
    with pytest.raises(RuntimeError, match="must be a SYCL XPU tensor"):
        torch.ops.fbgemm.all_to_one_device([tensor, torch.ones(2)], xpu)


@requires_two_xpus
def test_mixed_devices_keep_target_views_and_copy_contiguous():
    """In one call, a view already on the target comes back as the same view,
    and copies are contiguous whatever the input layout, as on CUDA."""
    source = torch.device("xpu:1")
    target = torch.device("xpu:0")
    on_target = torch.randn((10, 64), device=target)[:, :20]
    copied = [
        torch.arange(24, device=source).reshape(4, 6).t(),
        torch.randn((2, 3, 4, 5), device=source).to(memory_format=torch.channels_last),
        torch.randn((10, 64), device=source)[:, :20],
    ]
    outputs = torch.ops.fbgemm.all_to_one_device([on_target, *copied], target)
    assert outputs[0].data_ptr() == on_target.data_ptr()  # nosec B101
    assert outputs[0].stride() == on_target.stride()  # nosec B101
    for source_tensor, actual in zip(copied, outputs[1:]):
        assert actual.device == target  # nosec B101
        assert actual.is_contiguous()  # nosec B101
        torch.testing.assert_close(actual.cpu(), source_tensor.cpu())


@requires_two_xpus
def test_cross_device_stream_copy():
    """Copies in both directions on non-default streams."""
    for source_index, target_index in ((0, 1), (1, 0)):
        source = torch.device(f"xpu:{source_index}")
        target = torch.device(f"xpu:{target_index}")
        producer = torch.xpu.Stream(device=source)
        consumer = torch.xpu.Stream(device=target)
        with torch.xpu.stream(producer):
            inputs = [
                torch.arange(48, device=source).reshape(6, 8)[:, ::2],
                torch.full((3, 7), 2.5, dtype=torch.float16, device=source),
                torch.empty((0, 4), device=source),
            ]
        torch.xpu.current_stream(source).wait_stream(producer)
        with torch.xpu.stream(consumer):
            outputs = torch.ops.fbgemm.all_to_one_device(inputs, target)
            consumed = [tensor.clone() for tensor in outputs]
        consumer.synchronize()
        expected = [
            torch.arange(48).reshape(6, 8)[:, ::2],
            torch.full((3, 7), 2.5, dtype=torch.float16),
            torch.empty((0, 4)),
        ]
        for actual, reference in zip(consumed, expected):
            assert actual.device == target  # nosec B101
            torch.testing.assert_close(actual.cpu(), reference)
