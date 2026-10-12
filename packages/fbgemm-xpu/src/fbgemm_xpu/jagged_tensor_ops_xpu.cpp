/*
 * Copyright (c) Meta Platforms, Inc. and affiliates. All rights reserved.
 * Copyright (c) 2026 Intel Corporation. All Rights Reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */

////////////////////////////////////////////////////////////////////////////////
// SYCL PORT MAPPING TO FBGEMM CUDA SOURCE - JAGGED TENSOR OPS (HOST WRAPPERS)
////////////////////////////////////////////////////////////////////////////////
//
// XPU registrations for the jagged operators that CUDA binds in
// jagged_tensor_ops.cu to host functions from jagged_tensor_ops_autograd.cpp.
// They launch no kernel themselves: each one composes operators that already
// have XPU implementations, through the dispatcher.
//
// ORIGINAL CUDA SOURCE (v1.9.0):
//   Registration: fbgemm_gpu/src/jagged_tensor_ops/jagged_tensor_ops.cu
//   Functions:    fbgemm_gpu/src/jagged_tensor_ops/jagged_tensor_ops_autograd.cpp
//
// HOST FUNCTION MAPPING:
//   jagged_dense_elementwise_add_xpu → fbgemm_gpu::jagged_dense_elementwise_add
//   jagged_dense_elementwise_mul_xpu → fbgemm_gpu::jagged_dense_elementwise_mul
//
// AUTOGRAD FUNCTION MAPPING:
//   JaggedDenseMulXPUOp → JaggedDenseMulOp
//
// The upstream functions and JaggedDenseMulOp live in the fbgemm-gpu-cpu
// wheel, but they are internal C++ symbols rather than a supported cross-wheel
// API, so they are re-implemented here instead of linked against.
//
////////////////////////////////////////////////////////////////////////////////

#include <tuple>
#include <vector>

#include <ATen/ATen.h>
#include <ATen/core/dispatch/Dispatcher.h>
#include <c10/core/SymInt.h>
#include <c10/util/irange.h>
#include <torch/csrc/autograd/custom_function.h>
#include <torch/library.h>

namespace fbgemm_xpu {
namespace {

// Mirrors JaggedDenseMulOp: the forward and backward go through the
// fbgemm::jagged_dense_elementwise_mul_{forward,backward} operators, whose XPU
// kernels live in sycl_kernels/jagged_dense_elementwise_mul_{forward,backward}.
class JaggedDenseMulXPUOp
    : public torch::autograd::Function<JaggedDenseMulXPUOp> {
 public:
  static torch::autograd::variable_list forward(
      torch::autograd::AutogradContext* ctx,
      const at::Tensor& x_values,
      const std::vector<at::Tensor>& x_offsets,
      const at::Tensor& y) {
    std::vector<at::Tensor> tensors_to_save;
    tensors_to_save.reserve(2 + x_offsets.size());
    tensors_to_save.push_back(x_values);
    tensors_to_save.insert(
        tensors_to_save.end(), x_offsets.begin(), x_offsets.end());
    tensors_to_save.push_back(y);
    ctx->save_for_backward(tensors_to_save);

    static auto op = c10::Dispatcher::singleton()
                         .findSchemaOrThrow(
                             "fbgemm::jagged_dense_elementwise_mul_forward", "")
                         .typed<at::Tensor(
                             const at::Tensor& x_values,
                             const std::vector<at::Tensor>& x_offsets,
                             const at::Tensor& y)>();
    at::Tensor output = op.call(x_values, x_offsets, y);

    return {output};
  }

  static torch::autograd::variable_list backward(
      torch::autograd::AutogradContext* ctx,
      torch::autograd::variable_list grad_outputs) {
    const auto& saved_vars = ctx->get_saved_variables();
    const at::Tensor x_values = saved_vars.front();
    std::vector<at::Tensor> x_offsets;
    x_offsets.reserve(saved_vars.size() - 2);
    for (const auto i : c10::irange(1, saved_vars.size() - 1)) {
      x_offsets.push_back(saved_vars[i]);
    }
    const at::Tensor y = saved_vars.back();
    TORCH_CHECK(grad_outputs.size() == 1);

    static auto op =
        c10::Dispatcher::singleton()
            .findSchemaOrThrow(
                "fbgemm::jagged_dense_elementwise_mul_backward", "")
            .typed<std::tuple<at::Tensor, at::Tensor>(
                const at::Tensor& grad_output,
                const std::vector<at::Tensor>& x_offsets,
                const at::Tensor& y,
                const at::Tensor& x_values)>();
    auto outputs = op.call(grad_outputs[0], x_offsets, y, x_values);

    return {
        std::get<0>(outputs),
        torch::autograd::Variable(),  // x_offsets
        std::get<1>(outputs)};
  }
};

}  // namespace

// Output = x + y where x is jagged, y and output are dense. Mirrors
// fbgemm_gpu::jagged_dense_elementwise_add: x is densified with zero padding
// through fbgemm::jagged_to_padded_dense_forward, whose upstream Autograd
// kernel is the JaggedToPaddedDenseOp the CUDA function applies directly, and
// then added to y.
at::Tensor jagged_dense_elementwise_add_xpu(
    const at::Tensor& x_values,
    const std::vector<at::Tensor>& x_offsets,
    const at::Tensor& y) {
  // Construct max_lengths from y
  std::vector<c10::SymInt> max_lengths;
  max_lengths.reserve(x_offsets.size());
  for (const auto d : c10::irange(1, y.dim() - 1)) {
    max_lengths.push_back(y.sym_size(d));
  }
  TORCH_CHECK(max_lengths.size() == x_offsets.size());

  static auto op =
      c10::Dispatcher::singleton()
          .findSchemaOrThrow("fbgemm::jagged_to_padded_dense_forward", "")
          .typed<at::Tensor(
              const at::Tensor& values,
              const std::vector<at::Tensor>& offsets,
              at::ArrayRef<at::SymInt> max_lengths,
              const double padding_value)>();
  // Convert x to dense (assume padding is 0.0)
  auto xd = op.call(x_values, x_offsets, max_lengths, /*padding_value=*/0.0);

  auto dense_output = xd + y;
  return dense_output;
}

// Mirrors fbgemm_gpu::jagged_dense_elementwise_mul. The upstream Autograd-key
// registration handles differentiable calls; this XPU-key entry is reached when
// autograd is skipped, for example under torch.inference_mode().
std::tuple<at::Tensor, std::vector<at::Tensor>>
jagged_dense_elementwise_mul_xpu(
    const at::Tensor& x_values,
    const std::vector<at::Tensor>& x_offsets,
    const at::Tensor& y) {
  auto prod_values = JaggedDenseMulXPUOp::apply(x_values, x_offsets, y)[0];

  return {prod_values, x_offsets};
}

TORCH_LIBRARY_IMPL(fbgemm, XPU, m) {
  m.impl("jagged_dense_elementwise_add", &jagged_dense_elementwise_add_xpu);
  m.impl("jagged_dense_elementwise_mul", &jagged_dense_elementwise_mul_xpu);
}

}  // namespace fbgemm_xpu
