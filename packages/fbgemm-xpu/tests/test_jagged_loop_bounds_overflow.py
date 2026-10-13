# Copyright (c) 2026 Intel Corporation. All Rights Reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""int32 induction-overflow regressions for the jagged SYCL kernels.

Both jagged kernels index through 32-bit accessors and keep their loop counters
in int32. The ``numel < INT32_MAX`` gate in ``jagged_dense_elementwise_dense_output_``
bounds the *indices the kernels form*, not two intermediate values:

* the group-stride step ``counter += stride``, evaluated one past the last valid
  counter on the final active work-item, and
* ``iidx * 2 + 1`` in the paired inner loops, evaluated for the first ``iidx``
  that *fails* the loop test.

Both can pass INT32_MAX while every formed index stays in range. Signed overflow
is undefined behaviour, and in practice the wrapped-negative value keeps the loop
condition true, so the kernel takes one more iteration and writes through a
negative index. ``jagged_common.h`` therefore compares against ``limit - counter``
and against a precomputed pair count instead.

The sizes below are the smallest that make those two expressions overflow, and
each is asserted to do so at import time.

.. warning::
   These are crash regressions, not value regressions. The extra iteration
   writes *outside* the output buffer, so the buffer's own contents stay correct
   and detection relies on the out-of-bounds access faulting. All four were
   confirmed against a build with the guards reverted: each aborts the process
   ("Fatal Python error: Aborted") at the operator call. That means a
   regression takes the whole pytest process down rather than reporting a
   failed assertion, so run this file in its own pytest invocation. It also
   means detection is not guaranteed - a stray write that lands in mapped
   memory would go unnoticed.
"""

import fbgemm_xpu  # noqa: F401  - registers the fbgemm XPU operators
import pytest
import torch

pytestmark = pytest.mark.skipif(
    not torch.xpu.is_available(), reason="requires an XPU device"
)

_INT32_MAX = 2**31 - 1

# check_shape_and_partition_ geometry: threads_y is always
# kMaxThreads / kThreadGroupSize; threads_x is kThreadGroupSize when the inner
# dense dim is at least kThreadGroupSize / 2, and the inner dim itself below that.
_THREADS_Y = 1024 // 32
_THREADS_X_WIDE = 32
_THREADS_X_NARROW = 1  # inner dense dim of 1

_DTYPE = torch.bfloat16


def _div_round_up(a: int, b: int) -> int:
    return -(-a // b)


# --- large-outer: the group-stride step overflows -----------------------------
#
# With an inner dense dim of 1 the launch is threads_x = 1, threads_y = 32, and
# the work-group count is not capped (INT32_MAX / 32 groups are allowed, far
# more than are requested). The stride then exceeds the row count, so every
# work-item makes exactly one pass and the last active counter is limit - 1.
_LARGE_OUTER = 2**30 + 1
_OUTER_STRIDE = _div_round_up(_LARGE_OUTER, _THREADS_Y) * _THREADS_Y
assert _OUTER_STRIDE > _LARGE_OUTER, "stride must exceed the row count"  # nosec B101
# 1,073,741,824 + 1,073,741,856 = 2,147,483,680
assert (_LARGE_OUTER - 1) + _OUTER_STRIDE > _INT32_MAX, "step must overflow int32"  # nosec B101

# --- large-inner: iidx * 2 + 1 overflows -------------------------------------
#
# Kept just under INT32_MAX so the numel gate still selects the int32 kernel -
# that is the whole point, the gate holds and the arithmetic still overflows.
_LARGE_INNER = 2**31 - 2
assert _LARGE_INNER < _INT32_MAX, "must stay on the int32 indexing path"  # nosec B101
_INNER_PAIRS = _LARGE_INNER // 2
# First iidx that fails `iidx < inner_pairs`, reached from inner_begin = 0.
_FIRST_FAILING_IIDX = _div_round_up(_INNER_PAIRS, _THREADS_X_WIDE) * _THREADS_X_WIDE
# 2 * 2**30 + 1 = 2,147,483,649
assert 2 * _FIRST_FAILING_IIDX + 1 > _INT32_MAX, "product must overflow int32"  # nosec B101

# Padding is non-zero so the assertions on untouched regions also show that the
# kernel reached them, rather than passing on a zeroed buffer.
_PADDING = 3.0


@pytest.fixture(autouse=True)
def _release_device_memory():
    yield
    torch.xpu.empty_cache()


def _require_memory(elements: int) -> None:
    """Skip unless the device can hold `elements` elements plus 50% headroom."""
    needed = int(elements * _DTYPE.itemsize * 1.5)
    free, _ = torch.xpu.mem_get_info()
    if free < needed:
        pytest.skip(
            f"needs ~{needed / 2**30:.1f} GiB free on the device, "
            f"have {free / 2**30:.1f} GiB"
        )


def _uniform(t: torch.Tensor, value: float) -> bool:
    """True if every element equals `value`, without a full-size temporary."""
    return t.min().item() == value and t.max().item() == value


def test_dense_output_large_outer():
    """total_outer near 2**30 must not overflow the dense-output stride step."""
    # padded output only; `values` and `offsets` are negligible.
    _require_memory(_LARGE_OUTER)
    jagged_len = 8
    # 1-D values, so the inner dense dim is the folded 1 that makes threads_x 1.
    values = torch.arange(1, jagged_len + 1, device="xpu", dtype=_DTYPE)
    offsets = torch.tensor([0, jagged_len], device="xpu", dtype=torch.long)

    padded = torch.ops.fbgemm.jagged_to_padded_dense_forward(
        values, [offsets], [_LARGE_OUTER], _PADDING
    )

    assert padded.shape == (1, _LARGE_OUTER)  # nosec B101
    assert torch.equal(padded[0, :jagged_len], values)  # nosec B101
    assert _uniform(padded[0, jagged_len:], _PADDING)  # nosec B101


def test_jagged_output_large_outer():
    """nnz near 2**30 must not overflow the jagged-output stride step."""
    # `values` and `output`, both [nnz, 1].
    _require_memory(2 * _LARGE_OUTER)
    dense = torch.full((1, 1, 1), 7.0, device="xpu", dtype=_DTYPE)
    offsets = torch.tensor([0, 1], device="xpu", dtype=torch.long)

    output = torch.ops.fbgemm.dense_to_jagged_forward(
        dense, [offsets], _LARGE_OUTER
    )

    assert output.shape == (_LARGE_OUTER, 1)  # nosec B101
    # Row 0 gathers the dense value; every later row is past the jagged region
    # and gathers zero. Both halves are written, so this covers all nnz rows.
    assert output[0, 0].item() == 7.0  # nosec B101
    assert _uniform(output[1:], 0.0)  # nosec B101


def test_dense_output_large_inner():
    """An inner dense dim near INT32_MAX must not overflow iidx * 2 + 1."""
    # `values` and the padded output, both _LARGE_INNER elements.
    _require_memory(2 * _LARGE_INNER)
    values = torch.full((1, _LARGE_INNER), 5.0, device="xpu", dtype=_DTYPE)
    offsets = torch.tensor([0, 1], device="xpu", dtype=torch.long)

    padded = torch.ops.fbgemm.jagged_to_padded_dense_forward(
        values, [offsets], [1], _PADDING
    )

    assert padded.shape == (1, 1, _LARGE_INNER)  # nosec B101
    # The single row is inside the jagged region, so JaggedOpCopyX copies values
    # across the whole inner dim - including the final pair, which is where the
    # unguarded loop test would have wrapped.
    assert _uniform(padded[0, 0], 5.0)  # nosec B101


def test_jagged_output_large_inner():
    """The jagged-output paired inner loop has the same expression."""
    # `dense`, `values` and `output`, each _LARGE_INNER elements.
    _require_memory(3 * _LARGE_INNER)
    dense = torch.full((1, 1, _LARGE_INNER), 5.0, device="xpu", dtype=_DTYPE)
    offsets = torch.tensor([0, 1], device="xpu", dtype=torch.long)

    output = torch.ops.fbgemm.dense_to_jagged_forward(dense, [offsets], 1)

    assert output.shape == (1, _LARGE_INNER)  # nosec B101
    assert _uniform(output[0], 5.0)  # nosec B101
