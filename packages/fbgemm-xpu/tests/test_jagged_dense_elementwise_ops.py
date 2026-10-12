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
  write nothing.
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


@pytest.mark.parametrize("dtype", _DTYPES)
def test_mul_empty_dense_is_zero(dtype):
    """An empty dense operand gives x * 0, as on CPU, not uninitialized memory."""
    x_values, offsets, dense = _jagged_args(dtype, max_l=0)
    assert dense.numel() == 0 and x_values.numel() > 0  # nosec B101
    # Leave non-zero garbage in the caching allocator for the outputs to reuse.
    torch.full((64, _E), float("nan"), dtype=dtype, device="xpu")

    output = torch.ops.fbgemm.jagged_dense_elementwise_mul_forward(
        x_values.xpu(), [offsets.xpu()], dense.xpu()
    )
    x_values_grad, y_grad = torch.ops.fbgemm.jagged_dense_elementwise_mul_backward(
        x_values.xpu(), [offsets.xpu()], dense.xpu(), x_values.xpu()
    )

    assert torch.equal(output.cpu(), torch.zeros_like(x_values))  # nosec B101
    assert torch.equal(x_values_grad.cpu(), torch.zeros_like(x_values))  # nosec B101
    assert y_grad.shape == dense.shape  # nosec B101
