# Copyright (c) 2026 Intel Corporation. All Rights Reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""Mixed-device argument rejection for the jagged tensor operators.

The three shared host helpers in ``jagged_common.h`` hand every operand to a
kernel as a raw device pointer, but take their queue from the ambient device -
which the wrappers select with ``SYCL_DEVICE_GUARD`` from the *dense* operand
(``jagged_dense_elementwise_add_jagged_output.cpp``,
``dense_to_jagged_forward.cpp``). They used to validate only that ``x_values``
and each offsets tensor were "some XPU tensor": the dense operand and the
output were never checked, and no check compared device *indices*. A CPU dense
operand, or one on a second XPU, therefore selected the queue and then passed
pointers foreign to that queue's context - an invalid device-memory access
inside the kernel, surfacing asynchronously at whatever unrelated XPU operation
happened to synchronize next, instead of an error at the API boundary.

``check_tensors_on_same_xpu_`` now requires values, offsets, dense and output to
share one XPU device, before a queue is selected. Each case below is a clean
``RuntimeError``; the operators are exercised at both float16 (the FP16 fast
path, ``..._opt_``) and float32 (the generic kernel), which carry the check
separately. ``jagged_dense_elementwise_mul_backward`` also runs the check in its
jagged-jagged launcher, which takes the forward ``x_values`` as a second jagged
operand.

Every case keeps the operator's *first* tensor argument on an XPU: a CPU tensor
in that position routes the call to upstream's CPU implementation, which rejects
the XPU operands with its own message, so those combinations never reach the code
under test.

.. note::
   The check runs ahead of the ``y.numel() == 0`` early returns, so an empty
   dense operand on the wrong device is rejected rather than silently taking the
   early return - see ``test_add_rejects_empty_cpu_dense``. That is a deliberate
   behavior change: such a call previously returned a result.
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

_B = 2  # batch rows
_MAX_L = 2  # padded jagged length
_E = 8  # embedding dim; a multiple of 8 keeps rows 128-bit aligned
_TOTAL_L = 3  # jagged rows

# float16 takes the FP16 fast path, float32 the generic kernel.
_DTYPES = [torch.float16, torch.float32]

_REJECTED = "Not all tensors were on the same XPU"


def _jagged_args(dtype, max_l=_MAX_L):
    """Well-formed CPU arguments for the jagged/dense elementwise operators."""
    x_values = torch.arange(1, _TOTAL_L * _E + 1, dtype=dtype).view(_TOTAL_L, _E)
    offsets = torch.tensor([0, 1, _TOTAL_L], dtype=torch.int64)
    dense = torch.ones(_B, max_l, _E, dtype=dtype)
    return x_values, offsets, dense


@pytest.mark.parametrize("dtype", _DTYPES)
def test_add_rejects_cpu_dense(dtype):
    """A CPU dense operand cannot select the queue and pass a host pointer."""
    x_values, offsets, dense = _jagged_args(dtype)

    with pytest.raises(RuntimeError, match=_REJECTED):
        torch.ops.fbgemm.jagged_dense_elementwise_add_jagged_output(
            x_values.to("xpu"), [offsets.to("xpu")], dense
        )


@pytest.mark.parametrize("dtype", _DTYPES)
def test_add_rejects_cpu_offsets(dtype):
    """A CPU offsets tensor is rejected alongside the other operands."""
    x_values, offsets, dense = _jagged_args(dtype)

    with pytest.raises(RuntimeError, match=_REJECTED):
        torch.ops.fbgemm.jagged_dense_elementwise_add_jagged_output(
            x_values.to("xpu"), [offsets], dense.to("xpu")
        )


@pytest.mark.parametrize("dtype", _DTYPES)
def test_add_rejects_empty_cpu_dense(dtype):
    """An empty dense operand on the host is rejected, not early-returned."""
    x_values, offsets, dense = _jagged_args(dtype, max_l=0)
    assert dense.numel() == 0, "must reach the y.numel() == 0 early return"  # nosec B101

    with pytest.raises(RuntimeError, match=_REJECTED):
        torch.ops.fbgemm.jagged_dense_elementwise_add_jagged_output(
            x_values.to("xpu"), [offsets.to("xpu")], dense
        )


@requires_two_xpus
@pytest.mark.parametrize("dtype", _DTYPES)
def test_add_rejects_second_xpu_dense(dtype):
    """A dense operand on xpu:1 cannot pull xpu:0 pointers onto its queue."""
    x_values, offsets, dense = _jagged_args(dtype)

    with pytest.raises(RuntimeError, match=_REJECTED):
        torch.ops.fbgemm.jagged_dense_elementwise_add_jagged_output(
            x_values.to("xpu:0"), [offsets.to("xpu:0")], dense.to("xpu:1")
        )


@requires_two_xpus
@pytest.mark.parametrize("dtype", _DTYPES)
def test_add_rejects_second_xpu_offsets(dtype):
    """An offsets tensor on xpu:1 is rejected the same way."""
    x_values, offsets, dense = _jagged_args(dtype)

    with pytest.raises(RuntimeError, match=_REJECTED):
        torch.ops.fbgemm.jagged_dense_elementwise_add_jagged_output(
            x_values.to("xpu:0"), [offsets.to("xpu:1")], dense.to("xpu:0")
        )


@requires_two_xpus
@pytest.mark.parametrize("dtype", _DTYPES)
def test_add_accepts_second_xpu(dtype):
    """Arguments consistently on a non-zero XPU index still run."""
    x_values, offsets, dense = _jagged_args(dtype)

    expected, _ = torch.ops.fbgemm.jagged_dense_elementwise_add_jagged_output(
        x_values, [offsets], dense
    )
    actual, _ = torch.ops.fbgemm.jagged_dense_elementwise_add_jagged_output(
        x_values.to("xpu:1"), [offsets.to("xpu:1")], dense.to("xpu:1")
    )

    assert actual.device == torch.device("xpu:1")  # nosec B101
    assert torch.equal(actual.cpu(), expected)  # nosec B101


@pytest.mark.parametrize("dtype", _DTYPES)
def test_dense_to_jagged_rejects_cpu_offsets(dtype):
    """dense_to_jagged shares the helpers, so it rejects a CPU offsets tensor."""
    _, offsets, dense = _jagged_args(dtype)

    with pytest.raises(RuntimeError, match=_REJECTED):
        torch.ops.fbgemm.dense_to_jagged_forward(
            dense.to("xpu"), [offsets], _TOTAL_L
        )


@requires_two_xpus
@pytest.mark.parametrize("dtype", _DTYPES)
def test_dense_to_jagged_rejects_second_xpu_offsets(dtype):
    """dense_to_jagged rejects offsets on a different XPU than its dense."""
    _, offsets, dense = _jagged_args(dtype)

    with pytest.raises(RuntimeError, match=_REJECTED):
        torch.ops.fbgemm.dense_to_jagged_forward(
            dense.to("xpu:0"), [offsets.to("xpu:1")], _TOTAL_L
        )


def test_jagged_to_padded_dense_rejects_cpu_offsets():
    """The dense-output helper rejects a CPU offsets tensor."""
    x_values, offsets, _ = _jagged_args(torch.float32)

    with pytest.raises(RuntimeError, match=_REJECTED):
        torch.ops.fbgemm.jagged_to_padded_dense_forward(
            x_values.to("xpu"), [offsets], [_MAX_L], 0.0
        )


@requires_two_xpus
def test_jagged_to_padded_dense_rejects_second_xpu_offsets():
    """The dense-output helper compares device indices, not just device type."""
    x_values, offsets, _ = _jagged_args(torch.float32)

    with pytest.raises(RuntimeError, match=_REJECTED):
        torch.ops.fbgemm.jagged_to_padded_dense_forward(
            x_values.to("xpu:0"), [offsets.to("xpu:1")], [_MAX_L], 0.0
        )


@pytest.mark.parametrize("dtype", _DTYPES)
def test_mul_rejects_cpu_dense(dtype):
    """jagged_dense_elementwise_mul shares the jagged-output helpers."""
    x_values, offsets, dense = _jagged_args(dtype)

    with pytest.raises(RuntimeError, match=_REJECTED):
        torch.ops.fbgemm.jagged_dense_elementwise_mul(
            x_values.to("xpu"), [offsets.to("xpu")], dense
        )


@pytest.mark.parametrize("dtype", _DTYPES)
def test_mul_rejects_cpu_offsets(dtype):
    """A CPU offsets tensor is rejected by the multiply as well."""
    x_values, offsets, dense = _jagged_args(dtype)

    with pytest.raises(RuntimeError, match=_REJECTED):
        torch.ops.fbgemm.jagged_dense_elementwise_mul(
            x_values.to("xpu"), [offsets], dense.to("xpu")
        )


@pytest.mark.parametrize("dtype", _DTYPES)
def test_mul_rejects_empty_cpu_dense(dtype):
    """The zero-filled result for an empty dense operand comes after the check."""
    x_values, offsets, dense = _jagged_args(dtype, max_l=0)
    assert dense.numel() == 0, "must reach the y.numel() == 0 early return"  # nosec B101

    with pytest.raises(RuntimeError, match=_REJECTED):
        torch.ops.fbgemm.jagged_dense_elementwise_mul(
            x_values.to("xpu"), [offsets.to("xpu")], dense
        )


@pytest.mark.parametrize("dtype", _DTYPES)
def test_mul_backward_rejects_cpu_dense(dtype):
    """The x gradient's jagged-output launch rejects a CPU dense operand."""
    x_values, offsets, dense = _jagged_args(dtype)

    with pytest.raises(RuntimeError, match=_REJECTED):
        torch.ops.fbgemm.jagged_dense_elementwise_mul_backward(
            x_values.to("xpu"), [offsets.to("xpu")], dense, x_values.to("xpu")
        )


@pytest.mark.parametrize("dtype", _DTYPES)
def test_mul_backward_rejects_cpu_x_values(dtype):
    """The y gradient's jagged-jagged launch rejects CPU forward x_values."""
    x_values, offsets, dense = _jagged_args(dtype)

    with pytest.raises(RuntimeError, match=_REJECTED):
        torch.ops.fbgemm.jagged_dense_elementwise_mul_backward(
            x_values.to("xpu"), [offsets.to("xpu")], dense.to("xpu"), x_values
        )


@requires_two_xpus
@pytest.mark.parametrize("dtype", _DTYPES)
def test_mul_rejects_second_xpu_dense(dtype):
    """A dense operand on xpu:1 is rejected by the multiply."""
    x_values, offsets, dense = _jagged_args(dtype)

    with pytest.raises(RuntimeError, match=_REJECTED):
        torch.ops.fbgemm.jagged_dense_elementwise_mul(
            x_values.to("xpu:0"), [offsets.to("xpu:0")], dense.to("xpu:1")
        )


@requires_two_xpus
@pytest.mark.parametrize("dtype", _DTYPES)
def test_mul_backward_rejects_second_xpu_x_values(dtype):
    """The jagged-jagged launch compares device indices, not just device type."""
    x_values, offsets, dense = _jagged_args(dtype)

    with pytest.raises(RuntimeError, match=_REJECTED):
        torch.ops.fbgemm.jagged_dense_elementwise_mul_backward(
            x_values.to("xpu:0"),
            [offsets.to("xpu:0")],
            dense.to("xpu:0"),
            x_values.to("xpu:1"),
        )


def test_add_dense_output_rejects_cpu_offsets():
    """jagged_dense_elementwise_add densifies x through the dense-output helper."""
    x_values, offsets, dense = _jagged_args(torch.float32)

    with pytest.raises(RuntimeError, match=_REJECTED):
        torch.ops.fbgemm.jagged_dense_elementwise_add(
            x_values.to("xpu"), [offsets], dense.to("xpu")
        )


def test_add_dense_output_rejects_cpu_dense():
    """The dense operand reaches only the final ATen add, which rejects it."""
    x_values, offsets, dense = _jagged_args(torch.float32)

    with pytest.raises(RuntimeError, match="same device"):
        torch.ops.fbgemm.jagged_dense_elementwise_add(
            x_values.to("xpu"), [offsets.to("xpu")], dense
        )


@pytest.mark.parametrize("dtype", _DTYPES)
def test_xpu_usable_after_rejection(dtype):
    """Rejection happens before any launch, so the device stays usable."""
    x_values, offsets, dense = _jagged_args(dtype)

    with pytest.raises(RuntimeError, match=_REJECTED):
        torch.ops.fbgemm.jagged_dense_elementwise_add_jagged_output(
            x_values.to("xpu"), [offsets.to("xpu")], dense
        )

    expected, _ = torch.ops.fbgemm.jagged_dense_elementwise_add_jagged_output(
        x_values, [offsets], dense
    )
    actual, _ = torch.ops.fbgemm.jagged_dense_elementwise_add_jagged_output(
        x_values.to("xpu"), [offsets.to("xpu")], dense.to("xpu")
    )
    torch.xpu.synchronize()

    assert torch.equal(actual.cpu(), expected)  # nosec B101
