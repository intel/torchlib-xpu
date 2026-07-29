/*
 * Copyright (c) Meta Platforms, Inc. and affiliates. All rights reserved.
 * Copyright (c) 2026 Intel Corporation. All Rights Reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */

////////////////////////////////////////////////////////////////////////////////
// SYCL PORT MAPPING TO FBGEMM CUDA SOURCE - JAGGED TENSOR OPERATORS
////////////////////////////////////////////////////////////////////////////////
//
// Host-side operators, autograd functions (forward and backward) and XPU
// dispatch registrations for the jagged tensor family (dense_to_jagged,
// jagged_to_padded_dense, jagged_dense_elementwise_add_jagged_output and
// jagged_2d_to_dense), plus the jagged_to_padded_dense_backward gradient
// primitive. Device kernels are launched through the helpers declared in
// sycl_kernels/jagged_tensor_ops_kernels.h.
//
// ORIGINAL CUDA SOURCE (all paths relative to fbgemm_gpu/src/jagged_tensor_ops/):
//   Autograd / composites:
//     jagged_tensor_ops_autograd.cpp
//   Forward primitives:
//     dense_to_jagged_forward.cu
//     jagged_to_padded_dense_forward.cu
//   Backward primitive:
//     jagged_to_padded_dense_backward.cu
//   Operator schemas:
//     jagged_tensor_ops_cpu.cpp
//   CUDA dispatch registration for the composites (dense_to_jagged,
//   jagged_to_padded_dense, jagged_2d_to_dense):
//     jagged_tensor_ops.cu
//
// AUTOGRAD FUNCTION MAPPING (fbgemm_gpu namespace):
//   DenseToJaggedOp (SYCL)
//     -> DenseToJaggedOp (jagged_tensor_ops_autograd.cpp)
//   JaggedToPaddedDenseOp (SYCL)
//     -> JaggedToPaddedDenseOp (jagged_tensor_ops_autograd.cpp)
//   JaggedDenseAddJaggedOutputOp (SYCL)
//     -> JaggedDenseElementwiseAddJaggedOutOp (jagged_tensor_ops_autograd.cpp)
//
// OPERATOR / HOST FUNCTION MAPPING (fbgemm_gpu namespace):
//   dense_to_jagged (SYCL)
//     -> dense_to_jagged (jagged_tensor_ops_autograd.cpp)
//   dense_to_jagged_forward_xpu (SYCL)
//     -> dense_to_jagged_forward (dense_to_jagged_forward.cu)
//   jagged_to_padded_dense (SYCL)
//     -> jagged_to_padded_dense (jagged_tensor_ops_autograd.cpp)
//   jagged_to_padded_dense_forward_xpu (SYCL)
//     -> jagged_to_padded_dense_forward (jagged_to_padded_dense_forward.cu)
//   jagged_to_padded_dense_backward_xpu (SYCL)
//     -> jagged_to_padded_dense_backward (jagged_to_padded_dense_backward.cu)
//   jagged_dense_elementwise_add_jagged_output_xpu (SYCL)
//     -> jagged_dense_elementwise_add_jagged_output
//        (jagged_tensor_ops_autograd.cpp)
//   jagged_2d_to_dense_xpu (SYCL)
//     -> jagged_2d_to_dense (jagged_tensor_ops_autograd.cpp)
//
////////////////////////////////////////////////////////////////////////////////

#include <numeric>
#include <optional>
#include <tuple>
#include <vector>

#include <ATen/ATen.h>
#include <torch/all.h>
#include <torch/csrc/autograd/custom_function.h>
#include <torch/library.h>

#include "fbgemm_utils/tensor_utils.h"
#include "fbgemm_utils/utils.h"
#include "sycl_kernels/jagged_tensor_ops_kernels.h"

using at::Tensor;

namespace fbgemm_xpu {

// ============================================================================
// Forward host functions
// ============================================================================

Tensor dense_to_jagged_forward_xpu(
    const Tensor& dense,
    const std::vector<Tensor>& offsets,
    std::optional<at::SymInt> total_L) {
  TORCH_CHECK(dense.is_xpu(), "value must be a xpu tensor");
  for (auto& offset : offsets) {
    TORCH_CHECK(offset.is_xpu(), "offset must be a xpu tensor");
  }

  const int num_jagged_dim = dense.dim() - 2;
  TORCH_CHECK(
      offsets.size() == static_cast<size_t>(num_jagged_dim),
      "x_offsets.size(), ",
      offsets.size(),
      " != num_jagged_dim, ",
      num_jagged_dim);

  // D is the embedding dimension
  auto D = dense.size(-1);

  // If total_L is not given then compute it
  at::SymInt total_L_computed;
  if (total_L.has_value()) {
    total_L_computed = total_L.value();
  } else {
    total_L_computed = (int64_t)offsets.back().max().item<int64_t>();
  }
  auto values = at::empty_symint({total_L_computed, D}, dense.options());
  auto output = at::empty_like(values); // not used

  if (dense.numel() == 0 || values.numel() == 0) {
    return output;
  }

  SYCL_DEVICE_GUARD(dense);

  dense_to_jagged_forward_xpu_kernel(values, offsets, dense, output);

  return output;
}

Tensor jagged_to_padded_dense_forward_xpu(
    const Tensor& values,
    const std::vector<Tensor>& offsets,
    c10::SymIntArrayRef max_lengths,
    const double padding_value) {
  size_t num_jagged_dim = offsets.size();
  TORCH_CHECK(
      max_lengths.size() == num_jagged_dim,
      "max_lengths.size(), ",
      max_lengths.size(),
      " != num_jagged_dim, ",
      num_jagged_dim);

  TORCH_CHECK(values.is_xpu(), "value must be a xpu tensor");
  for (auto& offset : offsets) {
    TORCH_CHECK(offset.is_xpu(), "offset must be a xpu tensor");
  }

  SYCL_DEVICE_GUARD(values);

  const Tensor values_canonicalized = values.view(
      {values.size(0),
       std::accumulate(
           values.sizes().begin() + 1,
           values.sizes().end(),
           1,
           std::multiplies<size_t>())});
  at::SymDimVector padded_values_shape({at::SymInt(offsets[0].size(0) - 1)});
  padded_values_shape.insert(
      padded_values_shape.end(), max_lengths.begin(), max_lengths.end());

  // Canonicalize padded_values by unsqueeze the last dim if the inner dense
  // dimension is 1 and folded.
  const bool D_folded = values.dim() == 1;
  if (!D_folded) {
    padded_values_shape.push_back(values.size(-1));
  }
  Tensor padded_values =
      at::empty_symint(padded_values_shape, values.options());
  Tensor padded_values_view =
      D_folded ? padded_values.unsqueeze(-1) : padded_values;

  num_jagged_dim = padded_values_view.dim() - 2;
  TORCH_CHECK(
      offsets.size() == static_cast<size_t>(num_jagged_dim),
      "x_offsets.size(), ",
      offsets.size(),
      " != num_jagged_dim ",
      num_jagged_dim);

  if (padded_values_view.numel() == 0) {
    return padded_values;
  }

  jagged_to_padded_dense_forward_xpu_kernel(
      values_canonicalized,
      offsets,
      padded_values_view,
      padded_values_view,
      padding_value);

  return padded_values;
}

Tensor jagged_to_padded_dense_backward_xpu(
    const Tensor& grad_output,
    const std::vector<Tensor>& offsets,
    at::SymInt total_L) {
  auto grad_padded_values = grad_output;
  TORCH_CHECK(grad_padded_values.is_xpu(), "grad_output must be a xpu tensor");
  for (auto& offset : offsets) {
    TORCH_CHECK(offset.is_xpu(), "offset must be a xpu tensor");
  }

  SYCL_DEVICE_GUARD(grad_padded_values);

  // Canonicalize padded_values by unsqueeze the last dim if the inner dense
  // dimension is 1 and folded.
  const bool D_folded = grad_padded_values.dim() == offsets.size() + 1;
  Tensor grad_padded_values_view =
      D_folded ? grad_padded_values.unsqueeze(-1) : grad_padded_values;
  int32_t D = grad_padded_values_view.size(-1);

  // Initialize with zeros so output will be zero for the portion truncated
  // in forward.
  auto grad_values =
      at::zeros_symint({total_L, D}, grad_padded_values.options());

  if (grad_values.numel() != 0 && grad_padded_values.numel() != 0) {
    // The backward is a gather of the (dense) grad_output into the jagged
    // grad_values layout. This matches the semantics of the forward
    // dense_to_jagged kernel (SimpleRetSecondFunctor3 -> "return y"), so we
    // reuse it directly instead of adding a dedicated backward kernel.
    dense_to_jagged_forward_xpu_kernel(
        grad_values, offsets, grad_padded_values_view, grad_values);
  }

  return D_folded ? grad_values.squeeze(-1) : grad_values;
}

// ============================================================================
// Autograd functions
// ============================================================================

class DenseToJaggedOp : public torch::autograd::Function<DenseToJaggedOp> {
 public:
  static torch::autograd::variable_list forward(
      torch::autograd::AutogradContext* ctx,
      const Tensor& dense,
      const std::vector<Tensor>& offsets,
      const std::optional<at::SymInt>& total_L) {
    ctx->save_for_backward(offsets);

    // dims of dense tensor: <batch, [maxlen0, maxlen1, ...], embedding_dim>
    ctx->saved_data["dense_shape"] = dense.sym_sizes();

    static auto op =
        c10::Dispatcher::singleton()
            .findSchemaOrThrow("fbgemm::dense_to_jagged_forward", "")
            .typed<Tensor(
                const Tensor& dense,
                const std::vector<Tensor>& offsets,
                std::optional<at::SymInt> total_L)>();
    at::AutoDispatchBelowAutograd mode;
    auto output = op.call(dense, offsets, total_L);

    return {output};
  }

  static torch::autograd::variable_list backward(
      torch::autograd::AutogradContext* ctx,
      torch::autograd::variable_list grad_outputs) {
    auto offsets = ctx->get_saved_variables();
    auto dense_shape = ctx->saved_data["dense_shape"].toSymIntVector();
    TORCH_CHECK(grad_outputs.size() == 1);

    // The gradient wrt the dense input is the jagged grad scattered back into a
    // padded dense tensor whose shape matches the original dense input. This is
    // exactly jagged_to_padded_dense_forward with zero padding.
    static auto op =
        c10::Dispatcher::singleton()
            .findSchemaOrThrow("fbgemm::jagged_to_padded_dense_forward", "")
            .typed<Tensor(
                const Tensor& values,
                const std::vector<Tensor>& offsets,
                at::ArrayRef<at::SymInt> max_lengths,
                const double padding_value)>();
    auto dense_values_grad = op.call(
        grad_outputs[0],
        offsets,
        std::vector<at::SymInt>(dense_shape.begin() + 1, dense_shape.end() - 1),
        /*padding_value=*/0);

    TORCH_CHECK(dense_values_grad.sym_sizes() == c10::SymIntArrayRef(dense_shape));

    return {
        dense_values_grad,
        torch::autograd::Variable(), // offsets
        torch::autograd::Variable() // total_L
    };
  }
};

class JaggedToPaddedDenseOp
    : public torch::autograd::Function<JaggedToPaddedDenseOp> {
 public:
  static torch::autograd::variable_list forward(
      torch::autograd::AutogradContext* ctx,
      const Tensor& values,
      const std::vector<Tensor>& offsets,
      const c10::SymIntArrayRef max_lengths,
      const double padding_value) {
    ctx->save_for_backward(offsets);
    ctx->saved_data["total_L"] = values.sym_size(0);

    static auto op =
        c10::Dispatcher::singleton()
            .findSchemaOrThrow("fbgemm::jagged_to_padded_dense_forward", "")
            .typed<at::Tensor(
                const Tensor& values,
                const std::vector<Tensor>& offsets,
                at::ArrayRef<at::SymInt> max_lengths,
                const double padding_value)>();
    Tensor padded_values = op.call(values, offsets, max_lengths, padding_value);

    return {padded_values};
  }

  static torch::autograd::variable_list backward(
      torch::autograd::AutogradContext* ctx,
      torch::autograd::variable_list grad_outputs) {
    auto offsets = ctx->get_saved_variables();
    at::SymInt total_L = ctx->saved_data["total_L"].toSymInt();
    TORCH_CHECK(grad_outputs.size() == 1);

    TORCH_CHECK(total_L >= 0);
    static auto op =
        c10::Dispatcher::singleton()
            .findSchemaOrThrow("fbgemm::jagged_to_padded_dense_backward", "")
            .typed<at::Tensor(
                const Tensor& grad_output,
                const std::vector<Tensor>& offsets,
                at::SymInt total_L)>();
    auto grad_values = op.call(grad_outputs[0], offsets, total_L);

    return {
        grad_values,
        torch::autograd::Variable(), // offsets
        torch::autograd::Variable(), // max_lengths
        torch::autograd::Variable(), // padding_value
    };
  }
};

class JaggedDenseAddJaggedOutputOp
    : public torch::autograd::Function<JaggedDenseAddJaggedOutputOp> {
 public:
  static torch::autograd::variable_list forward(
      torch::autograd::AutogradContext* ctx,
      const Tensor& x_values,
      const std::vector<Tensor>& offsets,
      const Tensor& dense) {
    TORCH_CHECK(x_values.is_xpu(), "value must be a xpu tensor");
    for (auto& offset : offsets) {
      TORCH_CHECK(offset.is_xpu(), "offset must be a xpu tensor");
    }
    TORCH_CHECK(dense.is_xpu(), "dense must be a xpu tensor");

    ctx->save_for_backward(offsets);
    // dims of dense tensor: <batch, [maxlen0, maxlen1, ...], embedding_dim>
    ctx->saved_data["dense_shape"] = dense.sym_sizes();

    const int num_jagged_dim = dense.dim() - 2;
    TORCH_CHECK(
        offsets.size() == static_cast<size_t>(num_jagged_dim),
        "x_offsets.size(), ",
        offsets.size(),
        " != num_jagged_dim, ",
        num_jagged_dim);

    auto output = at::empty_like(x_values);
    if (dense.numel() == 0 || x_values.numel() == 0) {
      return {output};
    }

    SYCL_DEVICE_GUARD(dense);
    jagged_dense_elementwise_add_jagged_output_fwd_xpu_kn(
        x_values, offsets, dense, output);

    return {output};
  }

  static torch::autograd::variable_list backward(
      torch::autograd::AutogradContext* ctx,
      torch::autograd::variable_list grad_outputs) {
    auto offsets = ctx->get_saved_variables();
    TORCH_CHECK(grad_outputs.size() == 1);
    const auto& grad_output_tensor = grad_outputs[0];

    auto dense_shape = ctx->saved_data["dense_shape"].toSymIntVector();

    // output = x_values + dense (jagged output). The jagged input's gradient is
    // the incoming jagged grad (identity). The dense input's gradient is that
    // same grad scattered into a padded dense tensor of the original shape.
    static auto op =
        c10::Dispatcher::singleton()
            .findSchemaOrThrow("fbgemm::jagged_to_padded_dense_forward", "")
            .typed<Tensor(
                const Tensor& values,
                const std::vector<Tensor>& offsets,
                at::ArrayRef<at::SymInt> max_lengths,
                const double padding_value)>();

    auto dense_values_grad = op.call(
        grad_output_tensor,
        offsets,
        std::vector<at::SymInt>(dense_shape.begin() + 1, dense_shape.end() - 1),
        /*padding_value=*/0);

    TORCH_CHECK(
        dense_values_grad.sym_sizes() == c10::SymIntArrayRef(dense_shape));

    return {
        grad_output_tensor,
        torch::autograd::Variable(), // offsets
        dense_values_grad};
  }
};

// ============================================================================
// Top-Level Operator Functions
// ============================================================================

std::tuple<Tensor, std::vector<Tensor>> dense_to_jagged(
    const Tensor& dense,
    const std::vector<Tensor>& offsets,
    std::optional<at::SymInt> total_L) {
  return {DenseToJaggedOp::apply(dense, offsets, total_L)[0], offsets};
}

Tensor jagged_to_padded_dense(
    const Tensor& values,
    const std::vector<Tensor>& offsets,
    const c10::SymIntArrayRef max_lengths,
    const double padding_value) {
  return JaggedToPaddedDenseOp::apply(
      values, offsets, max_lengths, padding_value)[0];
}

std::tuple<Tensor, std::vector<Tensor>>
jagged_dense_elementwise_add_jagged_output_xpu(
    const Tensor& x_values,
    const std::vector<Tensor>& x_offsets,
    const Tensor& y) {
  auto sum_values =
      JaggedDenseAddJaggedOutputOp::apply(x_values, x_offsets, y)[0];

  return {sum_values, x_offsets};
}

// jagged_to_padded_dense_backward has no autograd wrapper: it is the gradient
// primitive invoked by JaggedToPaddedDenseOp::backward. It maps directly to the
// XPU host function above.

// jagged_2d_to_dense is a thin composite over jagged_to_padded_dense with a
// single jagged dimension and zero padding (mirrors the upstream FBGEMM CUDA
// implementation).
Tensor jagged_2d_to_dense_xpu(
    Tensor values,
    Tensor offsets,
    c10::SymInt max_sequence_length) {
  return jagged_to_padded_dense(
      values,
      {offsets},
      {std::move(max_sequence_length)},
      /*padding_value=*/0.0);
}

// ============================================================================
// PyTorch Operator Registration
// ============================================================================

// Device kernel primitives (no autograd graph): these are the forward/backward
// building blocks invoked from inside the autograd functions.
TORCH_LIBRARY_IMPL(fbgemm, XPU, m) {
  m.impl("dense_to_jagged_forward", &fbgemm_xpu::dense_to_jagged_forward_xpu);
  m.impl(
      "jagged_to_padded_dense_forward",
      &fbgemm_xpu::jagged_to_padded_dense_forward_xpu);
  m.impl(
      "jagged_to_padded_dense_backward",
      &fbgemm_xpu::jagged_to_padded_dense_backward_xpu);
}

// Autograd-aware operators: these build the autograd graph via
// torch::autograd::Function::apply and must be registered under the AutogradXPU
// key so the backward pass is wired up correctly.
TORCH_LIBRARY_IMPL(fbgemm, AutogradXPU, m) {
  m.impl("dense_to_jagged", &fbgemm_xpu::dense_to_jagged);
  m.impl("jagged_to_padded_dense", &fbgemm_xpu::jagged_to_padded_dense);
  m.impl(
      "jagged_dense_elementwise_add_jagged_output",
      &fbgemm_xpu::jagged_dense_elementwise_add_jagged_output_xpu);
  m.impl("jagged_2d_to_dense", &fbgemm_xpu::jagged_2d_to_dense_xpu);
}

} // namespace fbgemm_xpu
