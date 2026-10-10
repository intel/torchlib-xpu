# Copyright (c) 2026 Intel Corporation. All Rights Reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""Launch-size regression tests for the jagged SYCL kernels.

DPC++ is compiled with ``-fsycl-id-queries-fit-in-int``, so ``queue::submit``
throws "Provided range and/or offset does not fit in int" once the flattened
work-item count of a launch exceeds INT32_MAX. Two places cap the work-group
count against that limit: ``check_shape_and_partition_``, for the dense-output
path, and ``FBGEMM_XPU_JAGGED_OUTPUT_INVOKE_BODY``, which re-derives the count
from ``nnz``. Both rely on the kernel's group-stride loop to cover the rows the
smaller launch drops.

Which D reaches each cap follows from the inner extent. ``threads_x`` is
``ceil(D / 2)`` capped at ``kThreadGroupSize``, so it never exceeds D, and the
flattened count is close to the operand's ``numel``. Close, but not bounded by
it: the work-group count is rounded up to a whole multiple of ``threads_y``, so
the launch spans ``roundup(rows, threads_y) * threads_x`` work-items against a
``numel`` of ``rows * D``. The slack is up to ``(threads_y - 1) * threads_x``.

* **Jagged-output path** (``dense_to_jagged_forward``,
  ``jagged_dense_elementwise_add_jagged_output``,
  ``jagged_to_padded_dense_backward``) builds ``packed_accessor32``
  unconditionally, so ``numel`` cannot pass INT32_MAX - ATen raises first. Its
  cap is reachable anyway, but only at **D = 1**, the one width where
  ``threads_x == D`` and so the round-up slack is all that separates the launch
  from ``numel``: at ``nnz = 2**31 - 31``, ``numel`` is a legal 2147483617 while
  the uncapped launch is exactly 2**31. Every wider D has
  ``threads_x / D <= 1 / 2``, which leaves the launch a factor of two below a
  legal ``numel`` - far more than the slack can close. So the three tests below
  run at D = 1.
* **Dense-output path** (``jagged_to_padded_dense_forward``) falls back to
  ``packed_accessor64`` above the limit, so ``numel`` is unbounded and its cap
  fires at any D. ``test_jagged_to_padded_dense_forward_large_grid`` covers it at
  the smallest D that saturates ``threads_x``, which keeps the allocation as
  small as the limit allows.

The shape the jagged-output cap was first reported against is not the shape used
here. That report predates commit 5f977a5, which replaced CUDA's
``inner_dense_size >= kWarpSize / 2 ? kWarpSize : inner_dense_size`` with the
pair-derived ``ceil(D / 2)``:

    D = 16, nnz = 67,108,833, bf16
      CUDA geometry     threads_x = 32  ->  2**31 work-items   (submit threw)
      current geometry  threads_x =  8  ->  2**29 work-items   (4x under)

So the original shape runs clean today and would pass with or without the cap.
Reaching the limit at D = 16 now needs nnz ~ 268M, or ~16 GiB for ``x_values``
alone; D = 1 reproduces the same 2**31 launch in ~4 GiB, so that is what the
tests below use.

Each test is skipped unless the device has enough free memory; the smallest
shape that trips either limit needs multi-gigabyte operands. With
FBGEMM_XPU_TEST_SHARED_GPU=1, as CI sets it, a test is also skipped unless its
operands fit in half of the device's physical memory, leaving room for a second
test process on the same GPU.
"""

import os

import fbgemm_xpu  # noqa: F401  - registers the fbgemm XPU operators
import pytest
import torch

pytestmark = pytest.mark.skipif(
    not torch.xpu.is_available(), reason="requires an XPU device"
)

_DTYPE = torch.bfloat16

# Launch geometry check_shape_and_partition_ picks: threads_x is ceil(D / 2)
# capped at kThreadGroupSize - the inner loops walk two-element chunks - and
# threads_y is kMaxThreads / kThreadGroupSize. Derived rather than hardcoded so a
# geometry change fails these tests instead of quietly moving the launch back
# inside the int32 limit and testing nothing.
_THREADS_Y = 1024 // 32


def _threads_x(inner_dim: int) -> int:
    return min(32, -(-inner_dim // 2))


@pytest.fixture(autouse=True)
def _release_device_memory():
    yield
    torch.xpu.empty_cache()


def _require_memory(num_elements: int) -> None:
    """Skip unless the device can hold `num_elements` of the test dtype.

    The free figure covers this process's view of the device. With
    FBGEMM_XPU_TEST_SHARED_GPU=1 a cap at half of physical memory also covers
    what it cannot see: CI runs a second test process on the same GPU, and the
    driver backs allocations past physical memory with system memory instead
    of failing, so two over-committed processes crawl rather than raise.
    """
    needed = int(num_elements * _DTYPE.itemsize * 1.5)
    if os.environ.get("FBGEMM_XPU_TEST_SHARED_GPU") == "1":
        total = torch.xpu.get_device_properties().total_memory
        if needed > total // 2:
            pytest.skip(
                f"needs ~{needed / 2**30:.1f} GiB, over half of the device's "
                f"{total / 2**30:.1f} GiB, the limit while "
                f"FBGEMM_XPU_TEST_SHARED_GPU=1"
            )
    free, _ = torch.xpu.mem_get_info()
    if free < needed:
        pytest.skip(
            f"needs ~{needed / 2**30:.1f} GiB free on the device, "
            f"have {free / 2**30:.1f} GiB"
        )


# A whole-tensor reduction over the [nnz, D] tails below allocates a temporary
# the size of its input: `count_nonzero` takes 9 bytes per element (a bool mask
# plus an int64 accumulator) and `torch.equal` 1 byte. At this nnz that is up to
# 18 GiB - larger than the operands under test, and more than a 16 GiB device
# holds, so the assertion would OOM on a shape the operator itself handles in
# ~4 GiB. Reducing one slice at a time bounds the temporary to _CHUNK_ROWS,
# which keeps each test's peak at its operands plus ~64 MiB.
_CHUNK_ROWS = 1 << 26


def _assert_all_zero(actual: torch.Tensor) -> None:
    """Assert every element is zero, without a full-size temporary."""
    for start in range(0, actual.size(0), _CHUNK_ROWS):
        chunk = actual[start : start + _CHUNK_ROWS]
        assert not chunk.any(), (  # nosec B101
            f"non-zero element in rows [{start}, {start + chunk.size(0)})"
        )


def _assert_rows_equal(actual: torch.Tensor, expected: torch.Tensor) -> None:
    """Assert two row-aligned tensors match, without a full-size temporary."""
    assert actual.shape == expected.shape  # nosec B101
    for start in range(0, actual.size(0), _CHUNK_ROWS):
        stop = min(start + _CHUNK_ROWS, actual.size(0))
        assert torch.equal(actual[start:stop], expected[start:stop]), (  # nosec B101
            f"mismatch in rows [{start}, {stop})"
        )


# ============================================================================
# Dense-output path - cap reachable via the packed_accessor64 fallback
# ============================================================================

# D = 64 is the smallest convenient width where threads_x saturates at 32, which
# keeps outer * folded - and so the allocation - as small as the limit allows.
_OUT_D = 64
_OUT_MAX_L = 32
_OUT_THREADS_X = _threads_x(_OUT_D)
assert _OUT_THREADS_X == 32, "D must saturate threads_x or the cap is unreachable"  # nosec B101

# Largest work-group count whose flattened launch still fits in an int32_t, then
# the smallest outer * folded that asks for one more group than that. Uncapped,
# this shape produces exactly 2**31 work-items - one past the limit.
_OUT_MAX_BLOCKS = (2**31 - 1) // (_OUT_THREADS_X * _THREADS_Y)
_OUT_OUTER_FOLDED = (_OUT_MAX_BLOCKS + 1) * _THREADS_Y
_OUT_B = _OUT_OUTER_FOLDED // _OUT_MAX_L

# A distinctive padding value makes the coverage assertion real: the output
# buffer starts uninitialized, so an unvisited element cannot pass by holding a
# plausible zero.
_OUT_PADDING = -3.0

# One short jagged row in batch row 0; every other batch row is entirely padding.
_OUT_L = 4


def test_jagged_to_padded_dense_forward_large_grid():
    """A beyond-limit dense output is fully written by the group-stride loop."""
    _require_memory(_OUT_OUTER_FOLDED * _OUT_D)

    x_values = torch.arange(
        _OUT_L * _OUT_D, device="xpu", dtype=_DTYPE
    ).view(_OUT_L, _OUT_D)
    # offsets[1:] all equal _OUT_L, so batch row 0 holds the only jagged rows.
    offsets = torch.full(
        (_OUT_B + 1,), _OUT_L, device="xpu", dtype=torch.long
    )
    offsets[0] = 0

    output = torch.ops.fbgemm.jagged_to_padded_dense_forward(
        x_values, [offsets], [_OUT_MAX_L], _OUT_PADDING
    )

    assert output.shape == (_OUT_B, _OUT_MAX_L, _OUT_D)  # nosec B101
    assert torch.equal(output[0, :_OUT_L], x_values)  # nosec B101
    assert torch.equal(  # nosec B101
        output[0, _OUT_L:],
        torch.full_like(output[0, _OUT_L:], _OUT_PADDING),
    )
    # The capped launch covers _OUT_MAX_BLOCKS * _THREADS_Y flattened
    # (oidx, jidx) pairs on its first trip, leaving exactly _THREADS_Y - which is
    # precisely the final batch row's _OUT_MAX_L positions. So this row is
    # reached only by a second trip round the group-stride loop.
    assert _OUT_OUTER_FOLDED - _OUT_MAX_BLOCKS * _THREADS_Y == _OUT_MAX_L  # nosec B101
    assert torch.equal(  # nosec B101
        output[-1], torch.full_like(output[-1], _OUT_PADDING)
    )


# ============================================================================
# Jagged-output path - cap reachable at D = 1 via the work-group round-up
# ============================================================================

# D = 1 is the only width that reaches this cap: threads_x == D, so the launch
# exceeds numel by the work-group round-up alone. Any wider D halves threads_x
# relative to D and puts the launch out of reach of the int32 limit while numel
# is still legal - see the module docstring.
_D = 1
_THREADS_X = _threads_x(_D)
assert _THREADS_X == _D, "D must equal threads_x or the cap is unreachable"  # nosec B101

_MAX_BLOCKS = (2**31 - 1) // (_THREADS_X * _THREADS_Y)
_NNZ = _MAX_BLOCKS * _THREADS_Y + 1

# The launch these operators would issue uncapped, and the reason nothing rejects
# the shape first: the kernels build packed_accessor32 unconditionally, so an
# over-limit numel would raise in ATen before the submit. At D = 1 it does not.
assert -(-_NNZ // _THREADS_Y) * _THREADS_Y * _THREADS_X > 2**31 - 1  # nosec B101
assert _NNZ * _D <= 2**31 - 1  # nosec B101

# The jagged region is deliberately tiny: only nnz drives the grid, so a short
# dense operand keeps every allocation below to the [nnz, D] jagged tensors.
_LENGTHS = [4]


def _offsets() -> torch.Tensor:
    return torch.tensor([0, *_LENGTHS], device="xpu", dtype=torch.long)


def _dense() -> torch.Tensor:
    return torch.arange(
        len(_LENGTHS) * _LENGTHS[0] * _D, device="xpu", dtype=_DTYPE
    ).view(len(_LENGTHS), _LENGTHS[0], _D)


def test_jagged_to_padded_dense_backward_large_grid():
    """total_L past the int32 launch limit still scatters the dense gradient."""
    _require_memory(_NNZ * _D)
    dense = _dense()

    grad_values = torch.ops.fbgemm.jagged_to_padded_dense_backward(
        dense, [_offsets()], _NNZ
    )

    assert grad_values.shape == (_NNZ, _D)  # nosec B101
    # Rows inside the jagged region receive the dense gradient; the rest are the
    # zeros the operator pre-allocated for the portion truncated in forward.
    assert torch.equal(grad_values[: _LENGTHS[0]], dense[0])  # nosec B101
    _assert_all_zero(grad_values[_LENGTHS[0] :])


def test_jagged_dense_elementwise_add_jagged_output_large_grid():
    """Every row of a beyond-limit nnz is visited by the group-stride loop."""
    _require_memory(2 * _NNZ * _D)
    dense = _dense()
    # A uniform non-zero payload makes the assertion on the truncated rows a
    # coverage check: a row the launch failed to visit keeps whatever the
    # uninitialized output buffer held, not x + 0.
    x_values = torch.ones(_NNZ, _D, device="xpu", dtype=_DTYPE)

    output, _ = torch.ops.fbgemm.jagged_dense_elementwise_add_jagged_output(
        x_values, [_offsets()], dense
    )

    assert output.shape == (_NNZ, _D)  # nosec B101
    assert torch.equal(output[: _LENGTHS[0]], x_values[: _LENGTHS[0]] + dense[0])  # nosec B101
    # Rows past the jagged region add zero, so they must be exactly x. At this
    # nnz the cap drops a single row - the capped launch covers
    # _MAX_BLOCKS * _THREADS_Y == _NNZ - 1 of them - so the final row is
    # reached only by a second trip round the group-stride loop.
    _assert_rows_equal(output[_LENGTHS[0] :], x_values[_LENGTHS[0] :])


def test_dense_to_jagged_forward_large_grid():
    """total_L past the int32 launch limit still gathers the dense operand."""
    _require_memory(2 * _NNZ * _D)
    dense = _dense()

    output = torch.ops.fbgemm.dense_to_jagged_forward(dense, [_offsets()], _NNZ)

    assert output.shape == (_NNZ, _D)  # nosec B101
    assert torch.equal(output[: _LENGTHS[0]], dense[0])  # nosec B101
    # Rows past the jagged region gather zero. The output buffer starts
    # uninitialized, so this also checks the capped launch reached those rows.
    _assert_all_zero(output[_LENGTHS[0] :])
