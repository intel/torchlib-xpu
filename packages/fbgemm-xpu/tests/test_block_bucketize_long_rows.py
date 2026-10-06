# Copyright (c) 2026 Intel Corporation. All Rights Reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""Block bucketize on long rows, compared exactly with FBGEMM's CPU kernels.

Batch-1 inference (TorchRec row-wise sharding) bucketizes a few rows with
thousands of indices each. The XPU operator processes such rows in chunks
of one sub-group. Every case runs with each kernel path forced through
FBGEMM_XPU_BLOCK_BUCKETIZE_KERNEL and with the automatic choice, so the
paths must agree with the CPU kernels and with each other. Row lengths
straddle the chunk width (32) to cover partial and multiple chunks.
"""

import fbgemm_xpu  # noqa: F401  - registers the fbgemm XPU operators
import pytest
import torch

pytestmark = pytest.mark.skipif(
    not torch.xpu.is_available(), reason="requires an XPU device"
)

ENV = "FBGEMM_XPU_BLOCK_BUCKETIZE_KERNEL"
DEVICE = torch.device("xpu")
OUTPUTS = (
    "new_lengths",
    "new_indices",
    "new_weights",
    "new_pos",
    "unbucketize_permute",
    "bucket_mapping",
)


@pytest.fixture(params=["serial", "chunked", "auto"])
def kernel(request, monkeypatch):
    monkeypatch.setenv(ENV, request.param)
    return request.param


def _make_case(
    row_lengths,
    my_size,
    *,
    features=1,
    offset_dtype=torch.int64,
    index_dtype=torch.int64,
    weights_dtype=None,
    total_num_blocks=False,
    uneven=False,
    raw_ids=False,
    out_of_range=False,
    keep_orig_idx=False,
    keep_orig_idx_per_feature=False,
    batch_sizes=None,
    seed=0,
):
    """Inputs on CPU; row_lengths are feature-major, like KJT lengths."""
    generator = torch.Generator().manual_seed(seed)
    if batch_sizes is None:
        assert len(row_lengths) % features == 0  # nosec B101
        batch_sizes = [len(row_lengths) // features] * features
    assert sum(batch_sizes) == len(row_lengths)  # nosec B101

    lengths = torch.tensor(row_lengths, dtype=offset_dtype)
    global_num_blks = [2 * my_size if total_num_blocks else my_size] * features
    block_sizes = torch.tensor(
        [0 if raw_ids else 97 + 13 * t for t in range(features)], dtype=index_dtype
    )
    # Index range of feature t; with raw ids (block size 0) any value is valid.
    ranges = [
        int(block_sizes[t]) * global_num_blks[t] if not raw_ids else 50 * my_size
        for t in range(features)
    ]

    indices = []
    start = 0
    for t, batch in enumerate(batch_sizes):
        count = sum(row_lengths[start : start + batch])
        start += batch
        low, high = (0, ranges[t])
        if out_of_range:
            # Raw ids outside the table, negative only for signed index types.
            low, high = (-ranges[t], 3 * ranges[t])
        indices.append(torch.randint(low, high, (count,), generator=generator))
    indices = torch.cat(indices).to(index_dtype)

    kwargs = {"keep_orig_idx": keep_orig_idx}
    if weights_dtype is not None:
        kwargs["weights"] = torch.rand(
            indices.numel(), generator=generator, dtype=torch.float64
        ).to(weights_dtype)
    if total_num_blocks:
        kwargs["total_num_blocks"] = torch.tensor(global_num_blks, dtype=index_dtype)
    if uneven:
        # Ascending row offsets of my_size shards, last one at the range end;
        # the end is a multiple of the block count so CPU and XPU agree on
        # raw-id scaling.
        kwargs["block_bucketize_pos"] = [
            torch.tensor(
                [0]
                + sorted(
                    (torch.randperm(ranges[t] - 1, generator=generator)[: my_size - 1] + 1).tolist()
                )
                + [ranges[t]],
                dtype=index_dtype,
            )
            for t in range(features)
        ]
    if keep_orig_idx_per_feature:
        kwargs["keep_orig_idx_per_feature"] = torch.tensor(
            [t % 2 == 0 for t in range(features)], dtype=torch.bool
        )
    if len(set(batch_sizes)) > 1:
        kwargs["batch_size_per_feature"] = torch.tensor(batch_sizes, dtype=offset_dtype)
        kwargs["max_B"] = max(batch_sizes)
    return lengths, indices, block_sizes, kwargs


def _to(value, device):
    if isinstance(value, torch.Tensor):
        return value.to(device)
    if isinstance(value, list):
        return [item.to(device) for item in value]
    return value


def _call(op, lengths, indices, block_sizes, my_size, kwargs, device, **flags):
    return op(
        lengths.to(device),
        indices.to(device),
        block_sizes=block_sizes.to(device),
        my_size=my_size,
        **{key: _to(value, device) for key, value in kwargs.items()},
        **flags,
    )


def _check(case, my_size, *, inference=True, sequence=True, bucketize_pos=True):
    lengths, indices, block_sizes, kwargs = case
    if inference:
        op = torch.ops.fbgemm.block_bucketize_sparse_features_inference
        flags = {"return_bucket_mapping": True}
    else:
        op = torch.ops.fbgemm.block_bucketize_sparse_features
        flags = {}
    flags.update(sequence=sequence, bucketize_pos=bucketize_pos)
    expected = _call(op, lengths, indices, block_sizes, my_size, kwargs, "cpu", **flags)
    actual = _call(op, lengths, indices, block_sizes, my_size, kwargs, DEVICE, **flags)
    assert len(actual) == len(expected)  # nosec B101
    for name, got, want in zip(OUTPUTS, actual, expected):
        if name == "bucket_mapping" and not sequence:
            # Only sequence=True fills it; CPU and CUDA leave it uninitialized.
            continue
        assert (got is None) == (want is None), name  # nosec B101
        if want is not None:
            torch.testing.assert_close(
                got.cpu(), want, rtol=0, atol=0, msg=lambda message, name=name: f"{name}: {message}"
            )
    return actual


ROW_LENGTHS = {
    "empty": [0],
    "one": [1],
    "below_chunk": [31],
    "one_chunk": [32],
    "above_chunk": [33],
    "long": [1000],
    "dlrm_v3": [16384],
    "mixed": [5000, 0, 33, 1, 64, 2000],
}


@pytest.mark.parametrize("rows", ROW_LENGTHS.values(), ids=ROW_LENGTHS.keys())
@pytest.mark.parametrize("my_size", [1, 2, 7, 16, 31, 32, 33, 64])
@pytest.mark.parametrize("sequence", [True, False])
def test_row_lengths(kernel, rows, my_size, sequence):
    _check(_make_case(rows, my_size, seed=my_size), my_size, sequence=sequence)


@pytest.mark.parametrize("offset_dtype", [torch.int32, torch.int64])
@pytest.mark.parametrize("index_dtype", [torch.int32, torch.int64])
@pytest.mark.parametrize("sequence", [True, False])
def test_dtypes(kernel, offset_dtype, index_dtype, sequence):
    case = _make_case(
        [3000, 7, 0, 129],
        features=2,
        my_size=8,
        offset_dtype=offset_dtype,
        index_dtype=index_dtype,
        weights_dtype=torch.float,
    )
    _check(case, 8, sequence=sequence)


OPTIONS = {
    "weights_float": {"weights_dtype": torch.float},
    "weights_double": {"weights_dtype": torch.double},
    "keep_orig_idx": {"keep_orig_idx": True},
    "keep_orig_idx_per_feature": {"keep_orig_idx_per_feature": True},
    "total_num_blocks": {"total_num_blocks": True},
    "uneven": {"uneven": True},
    "uneven_raw_ids": {"uneven": True, "raw_ids": True},
    "raw_ids": {"raw_ids": True},
    "out_of_range": {"out_of_range": True},
    "out_of_range_total_num_blocks": {"out_of_range": True, "total_num_blocks": True},
    "variable_batch": {"batch_sizes": [1, 3, 2]},
}


@pytest.mark.parametrize("options", OPTIONS.values(), ids=OPTIONS.keys())
@pytest.mark.parametrize("sequence", [True, False])
@pytest.mark.parametrize("bucketize_pos", [True, False])
def test_options(kernel, options, sequence, bucketize_pos):
    rows = [4000, 33, 0, 1500, 64, 2]
    case = _make_case(rows, 16, features=3, seed=7, **options)
    _check(case, 16, sequence=sequence, bucketize_pos=bucketize_pos)


@pytest.mark.parametrize("sequence", [True, False])
@pytest.mark.parametrize("weights_dtype", [None, torch.float])
def test_training_operator(kernel, sequence, weights_dtype):
    case = _make_case([2500, 31, 700, 0], 8, features=2, weights_dtype=weights_dtype)
    _check(case, 8, inference=False, sequence=sequence)


def test_populate_bucketized_permute_matches_unbucketize_permute(kernel):
    lengths, indices, block_sizes, kwargs = case = _make_case([6000, 45, 0, 300], 16, features=2)
    actual = _check(case, 16, sequence=True)
    permute = torch.ops.fbgemm.populate_bucketized_permute(
        lengths.to(DEVICE), actual[0], actual[5]
    )
    torch.testing.assert_close(permute.cpu(), actual[4].cpu(), rtol=0, atol=0)


def test_random_configurations(kernel):
    generator = torch.Generator().manual_seed(1234)

    def choice(values):
        return values[int(torch.randint(len(values), (1,), generator=generator))]

    def chance(probability):
        return bool(torch.rand(1, generator=generator) < probability)

    for seed in range(40):
        features = choice([1, 2, 3, 4])
        batch = choice([1, 2, 3, 4, 5, 6])
        rows = [choice([0, 1, 31, 32, 33, 300, 3000]) for _ in range(features * batch)]
        options = {
            "weights_dtype": choice([None, torch.float, torch.double]),
            "total_num_blocks": chance(0.3),
            "uneven": chance(0.3),
            "keep_orig_idx": chance(0.2),
            "keep_orig_idx_per_feature": chance(0.2),
            "out_of_range": chance(0.3),
            "offset_dtype": choice([torch.int32, torch.int64]),
            "index_dtype": choice([torch.int32, torch.int64]),
        }
        my_size = choice([1, 2, 3, 8, 16, 31, 32, 40])
        case = _make_case(rows, my_size, features=features, seed=seed, **options)
        _check(case, my_size, sequence=chance(0.5), bucketize_pos=chance(0.5))


def test_rejects_unknown_kernel(monkeypatch):
    monkeypatch.setenv(ENV, "fast")
    lengths, indices, block_sizes, kwargs = _make_case([64], 2)
    with pytest.raises(RuntimeError, match=ENV):
        _call(
            torch.ops.fbgemm.block_bucketize_sparse_features_inference,
            lengths, indices, block_sizes, 2, kwargs, DEVICE,
            sequence=True, bucketize_pos=False,
        )
