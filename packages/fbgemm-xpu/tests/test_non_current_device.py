# Copyright (c) 2026 Intel Corporation. All Rights Reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""Sparse operators must run on the queue of their tensors' device.

Inputs live on ``xpu:1`` while ``xpu:0`` stays current, and they become valid
only after a slow kernel chain on ``xpu:1``. A kernel submitted to the queue of
the current device instead would read them before they are written. Results are
compared with FBGEMM's CPU kernels. TorchRec row-wise sharding calls these
operators with its output device, which need not be the current one.
"""

import fbgemm_xpu  # noqa: F401  - registers the fbgemm XPU operators
import pytest
import torch

requires_two_xpus = pytest.mark.skipif(
    torch.xpu.is_available() and torch.xpu.device_count() < 2,
    reason="requires two XPU devices",
)

pytestmark = [
    pytest.mark.skipif(not torch.xpu.is_available(), reason="requires an XPU device"),
    requires_two_xpus,
]

ITERATIONS = 5
SAME_DEVICE_ERROR = "Not all tensors were on the same XPU"


@pytest.fixture
def devices():
    torch.xpu.set_device(0)
    return torch.device("xpu:0"), torch.device("xpu:1")


def _late(tensors, device):
    """Return copies on device that become valid only after slow work there.

    The copies from the CPU block, so they are made before the slow work is
    queued; made after it, they would wait for it and leave nothing pending.
    """
    on_device = [tensor.to(device) if tensor is not None else None for tensor in tensors]
    weight = torch.randn((2048, 2048), device=device)
    slow = weight
    for _ in range(8):
        slow = slow @ weight
    zero = slow[0, 0] * 0
    return [
        tensor + zero.to(tensor.dtype) if tensor is not None else None
        for tensor in on_device
    ]


def _assert_same(actual, expected):
    if isinstance(expected, (list, tuple)):
        assert len(actual) == len(expected)  # nosec B101
        for actual_item, expected_item in zip(actual, expected):
            _assert_same(actual_item, expected_item)
    elif expected is None:
        assert actual is None  # nosec B101
    else:
        torch.testing.assert_close(actual.cpu(), expected, rtol=0, atol=0)


def _run(operator, inputs, devices, *args):
    current, data = devices
    expected = operator(*inputs, *args)
    for _ in range(ITERATIONS):
        actual = operator(*_late(inputs, data), *args)
        torch.xpu.synchronize(data)
        torch.xpu.synchronize(current)
        assert torch.xpu.current_device() == current.index  # nosec B101
        _assert_same(actual, expected)


def _sparse_features():
    generator = torch.Generator().manual_seed(0)
    lengths = torch.randint(0, 9, (12,), generator=generator, dtype=torch.int32)
    indices = torch.randint(0, 1000, (int(lengths.sum()),), generator=generator, dtype=torch.int32)
    return lengths, indices


def test_block_bucketize_sparse_features_inference(devices):
    lengths, indices = _sparse_features()
    block_sizes = torch.full((3,), 250, dtype=torch.int32)

    def operator(lengths, indices, block_sizes):
        return torch.ops.fbgemm.block_bucketize_sparse_features_inference(
            lengths, indices, False, True, block_sizes, 4, None,
            return_bucket_mapping=True,
        )

    _run(operator, [lengths, indices, block_sizes], devices)


def test_block_bucketize_sparse_features(devices):
    lengths, indices = _sparse_features()
    block_sizes = torch.full((3,), 250, dtype=torch.int32)

    def operator(lengths, indices, block_sizes):
        return torch.ops.fbgemm.block_bucketize_sparse_features(
            lengths, indices, False, True, block_sizes, 4, None
        )

    _run(operator, [lengths, indices, block_sizes], devices)


def test_populate_bucketized_permute(devices):
    lengths, indices = _sparse_features()
    block_sizes = torch.full((3,), 250, dtype=torch.int32)
    bucketized = torch.ops.fbgemm.block_bucketize_sparse_features_inference(
        lengths, indices, False, True, block_sizes, 4, None, return_bucket_mapping=True
    )
    _run(
        torch.ops.fbgemm.populate_bucketized_permute,
        [lengths, bucketized[0], bucketized[5]],
        devices,
    )


@pytest.mark.parametrize("name", ["permute_2D_sparse_data", "permute_1D_sparse_data"])
def test_permute_sparse_data(devices, name):
    lengths, indices = _sparse_features()
    if name == "permute_2D_sparse_data":
        lengths = lengths.view(3, 4)
    permute = torch.tensor([2, 0, 1], dtype=torch.int32)
    if name == "permute_1D_sparse_data":
        permute = torch.randperm(12, generator=torch.Generator().manual_seed(1)).int()
    _run(getattr(torch.ops.fbgemm, name), [permute, lengths, indices], devices)


def test_invert_permute(devices):
    permute = torch.randperm(1000, generator=torch.Generator().manual_seed(2)).int()
    _run(torch.ops.fbgemm.invert_permute, [permute], devices)


def test_expand_into_jagged_permute(devices):
    permute = torch.tensor([2, 0, 1], dtype=torch.int32)
    lengths = torch.tensor([3, 5, 2], dtype=torch.int32)
    input_offsets = torch.ops.fbgemm.asynchronous_complete_cumsum(lengths)
    output_offsets = torch.ops.fbgemm.asynchronous_complete_cumsum(lengths[permute.long()])
    _run(
        torch.ops.fbgemm.expand_into_jagged_permute,
        [permute, input_offsets, output_offsets],
        devices,
        int(lengths.sum()),
    )


def test_jagged_index_select_2d_forward(devices):
    generator = torch.Generator().manual_seed(3)
    lengths = torch.tensor([3, 0, 5, 2], dtype=torch.int64)
    values = torch.randn((int(lengths.sum()), 16), generator=generator)
    indices = torch.tensor([2, 0, 3, 2], dtype=torch.int64)
    input_offsets = torch.ops.fbgemm.asynchronous_complete_cumsum(lengths)[1:]
    output_offsets = torch.ops.fbgemm.asynchronous_complete_cumsum(lengths[indices])[1:]
    _run(
        torch.ops.fbgemm.jagged_index_select_2d_forward,
        [values, indices, input_offsets, output_offsets],
        devices,
        int(output_offsets[-1]),
    )


def test_block_bucketize_rejects_mixed_devices(devices):
    current, data = devices
    lengths, indices = _sparse_features()
    lengths = lengths.to(data)
    indices = indices.to(current)
    block_sizes = torch.full((3,), 250, dtype=torch.int32, device=data)

    with pytest.raises(RuntimeError, match=SAME_DEVICE_ERROR):
        torch.ops.fbgemm.block_bucketize_sparse_features_inference(
            lengths,
            indices,
            False,
            True,
            block_sizes,
            4,
            None,
            return_bucket_mapping=True,
        )


def test_populate_bucketized_permute_rejects_mixed_devices(devices):
    current, data = devices
    lengths = torch.tensor([1, 2, 0], dtype=torch.int32, device=data)
    bucketized_lengths = torch.zeros(12, dtype=torch.int32, device=current)
    bucket_mapping = torch.zeros(3, dtype=torch.int32, device=data)

    with pytest.raises(RuntimeError, match=SAME_DEVICE_ERROR):
        torch.ops.fbgemm.populate_bucketized_permute(
            lengths, bucketized_lengths, bucket_mapping
        )


@pytest.mark.parametrize("name", ["permute_2D_sparse_data", "permute_1D_sparse_data"])
@pytest.mark.parametrize("mismatch", ["required", "optional_weights"])
def test_permute_sparse_data_rejects_mixed_devices(devices, name, mismatch):
    current, data = devices
    lengths, indices = _sparse_features()
    if name == "permute_2D_sparse_data":
        lengths = lengths.view(3, 4)
        permute = torch.tensor([2, 0, 1], dtype=torch.int32)
    else:
        permute = torch.randperm(
            12, generator=torch.Generator().manual_seed(1)
        ).int()

    permute = permute.to(current if mismatch == "required" else data)
    lengths = lengths.to(data)
    indices = indices.to(data)
    weights = torch.randn(indices.numel(), device=current)
    if mismatch == "required":
        weights = weights.to(data)

    with pytest.raises(RuntimeError, match=SAME_DEVICE_ERROR):
        getattr(torch.ops.fbgemm, name)(permute, lengths, indices, weights)


def test_jagged_index_select_rejects_mixed_devices(devices):
    current, data = devices
    values = torch.randn((4, 8), device=data)
    indices = torch.tensor([0], dtype=torch.int64, device=current)
    input_offsets = torch.tensor([4], dtype=torch.int64, device=data)
    output_offsets = torch.tensor([4], dtype=torch.int64, device=data)

    with pytest.raises(RuntimeError, match=SAME_DEVICE_ERROR):
        torch.ops.fbgemm.jagged_index_select_2d_forward(
            values, indices, input_offsets, output_offsets, 4
        )
