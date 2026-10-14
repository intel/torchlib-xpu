# Copyright (c) 2026 Intel Corporation. All Rights Reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""XPU-specific checks for jagged_dense_elementwise_add and _mul.

FBGEMM's ``jagged/elementwise_binary_test.py`` (enabled through the shared test
patch) owns the forward and float32 gradcheck parity of both operators. This
module covers what it does not reach:

* the XPU dispatch-key entries of ``jagged_dense_elementwise_add`` and
  ``jagged_dense_elementwise_mul``. Upstream tests always enter ``_mul`` through
  its Autograd-key registration; ``torch.inference_mode()`` skips that key and
  calls the XPU entry directly. ``_add`` has no Autograd-key registration and
  always runs its XPU entry;
* the ``_mul`` backward kernels at float16 and bfloat16, which gradcheck never
  runs (it upcasts to float32) and the CPU backward cannot compute for bfloat16;
* the zero-filled result for an empty dense operand, where the CUDA kernels
  write nothing;
* the shape check on the forward ``x_values`` passed to the ``_mul`` backward,
  which CPU and CUDA read out of bounds when it is smaller than ``grad_output``.
"""

import fbgemm_xpu  # noqa: F401  - registers the fbgemm XPU operators
import pytest
import torch

pytestmark = pytest.mark.skipif(
    not torch.xpu.is_available(), reason="requires an XPU device"
)

_OPERATORS = [
    "jagged_dense_elementwise_add",
    "jagged_dense_elementwise_mul",
    "jagged_dense_elementwise_mul_forward",
    "jagged_dense_elementwise_mul_backward",
]

_DTYPES = [torch.float32, torch.float16, torch.bfloat16]

# float16 rows of 8 elements are 128-bit aligned, so the forward takes the FP16
# fast path; the other dtypes take the generic kernel.
_E = 8
_MAX_L = 2


@pytest.fixture
def fill_uninitialized_memory():
    """Make an unwritten output observable as NaN instead of stale bytes."""
    deterministic = torch.are_deterministic_algorithms_enabled()
    fill = torch.utils.deterministic.fill_uninitialized_memory
    torch.use_deterministic_algorithms(True)
    torch.utils.deterministic.fill_uninitialized_memory = True
    yield
    torch.utils.deterministic.fill_uninitialized_memory = fill
    torch.use_deterministic_algorithms(deterministic)


def _jagged_args(dtype, max_l=_MAX_L):
    """CPU arguments whose second batch row is truncated by max_l."""
    generator = torch.Generator().manual_seed(0)
    offsets = torch.tensor([0, 1, 4], dtype=torch.int64)
    x_values = torch.randn((4, _E), generator=generator).to(dtype)
    dense = torch.randn((2, max_l, _E), generator=generator).to(dtype)
    return x_values, offsets, dense


def _call(name, x_values, offsets, dense):
    if name == "add":
        return torch.ops.fbgemm.jagged_dense_elementwise_add(
            x_values, [offsets], dense
        )
    output, _ = torch.ops.fbgemm.jagged_dense_elementwise_mul(
        x_values, [offsets], dense
    )
    return output


@pytest.mark.parametrize("name", _OPERATORS)
def test_xpu_kernel_registered(name):
    assert torch._C._dispatch_has_kernel_for_dispatch_key(  # nosec B101
        f"fbgemm::{name}", "XPU"
    )


@pytest.mark.parametrize("dtype", _DTYPES)
@pytest.mark.parametrize("name", ["add", "mul"])
def test_inference_mode_matches_cpu(name, dtype):
    x_values, offsets, dense = _jagged_args(dtype)
    with torch.inference_mode():
        expected = _call(name, x_values, offsets, dense)
        actual = _call(name, x_values.xpu(), offsets.xpu(), dense.xpu())

    assert actual.device.type == "xpu"  # nosec B101
    torch.testing.assert_close(actual.cpu(), expected, rtol=0, atol=0)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_mul_backward_low_precision(dtype):
    """Each gradient element is one product rounded once, so it is exact."""
    x_values, offsets, dense = _jagged_args(dtype)
    grad = torch.randn((x_values.size(0), _E), generator=torch.Generator().manual_seed(1)).to(dtype)

    x = x_values.xpu().requires_grad_()
    y = dense.xpu().requires_grad_()
    output, _ = torch.ops.fbgemm.jagged_dense_elementwise_mul(x, [offsets.xpu()], y)
    output.backward(grad.xpu())

    # The product of two float16 or bfloat16 values is exact in float32, so the
    # float32 reference rounded to dtype is the correctly rounded gradient.
    x_ref = x_values.float().requires_grad_()
    y_ref = dense.float().requires_grad_()
    output_ref, _ = torch.ops.fbgemm.jagged_dense_elementwise_mul(
        x_ref, [offsets], y_ref
    )
    output_ref.backward(grad.float())

    assert torch.equal(x.grad.cpu(), x_ref.grad.to(dtype))  # nosec B101
    assert torch.equal(y.grad.cpu(), y_ref.grad.to(dtype))  # nosec B101


@pytest.mark.usefixtures("fill_uninitialized_memory")
@pytest.mark.parametrize("dtype", _DTYPES)
def test_mul_empty_dense_is_zero(dtype):
    """An empty dense operand gives x * 0, as on CPU, not uninitialized memory."""
    x_values, offsets, dense = _jagged_args(dtype, max_l=0)
    assert dense.numel() == 0 and x_values.numel() > 0  # nosec B101

    output = torch.ops.fbgemm.jagged_dense_elementwise_mul_forward(
        x_values.xpu(), [offsets.xpu()], dense.xpu()
    )
    x_values_grad, y_grad = torch.ops.fbgemm.jagged_dense_elementwise_mul_backward(
        x_values.xpu(), [offsets.xpu()], dense.xpu(), x_values.xpu()
    )

    assert torch.equal(output.cpu(), torch.zeros_like(x_values))  # nosec B101
    assert torch.equal(x_values_grad.cpu(), torch.zeros_like(x_values))  # nosec B101
    assert y_grad.shape == dense.shape  # nosec B101


# The offsets place jagged rows 0-2 inside the dense tensor, so both shapes
# would make the y gradient read past the end of x_values.
@pytest.mark.parametrize(
    "shape", [(2, _E), (4, _E - 1)], ids=["fewer_rows", "fewer_columns"]
)
def test_mul_backward_rejects_mismatched_x_values(shape):
    grad_output, offsets, dense = _jagged_args(torch.float32)
    x_values = torch.ones(shape)

    with pytest.raises(
        RuntimeError, match=r"x_values\.sizes\(\), .* != grad_output\.sizes\(\)"
    ):
        torch.ops.fbgemm.jagged_dense_elementwise_mul_backward(
            grad_output.xpu(), [offsets.xpu()], dense.xpu(), x_values.xpu()
        )
