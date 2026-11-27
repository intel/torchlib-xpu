/*
 * Copyright (c) Meta Platforms, Inc. and affiliates. All rights reserved.
 * Copyright (c) 2026 Intel Corporation. All Rights Reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */

#include <ATen/Dispatch.h>
#include <torch/csrc/autograd/custom_function.h>

#include "fbgemm_utils/utils.h"
#include "jagged_common.h"
#include "jagged_dense_elementwise_add_jagged_output.h"
#include "jagged_to_padded_dense_forward.h"

namespace fbgemm_xpu {

namespace {

////////////////////////////////////////////////////////////////////////////////
// JaggedDenseAddJaggedOutputXPUOp - Autograd Function
////////////////////////////////////////////////////////////////////////////////
//
// CUDA SOURCE MAPPING:
//   CUDA Class: JaggedDenseAddJaggedOutputGPUOp
//   CUDA File: fbgemm_gpu/src/jagged_tensor_ops/jagged_to_padded_dense_forward.cu
//
////////////////////////////////////////////////////////////////////////////////
class JaggedDenseAddJaggedOutputXPUOp
    : public torch::autograd::Function<JaggedDenseAddJaggedOutputXPUOp> {
public:
    static torch::autograd::variable_list forward(
        torch::autograd::AutogradContext* ctx,
        const at::Tensor& x_values,
        const std::vector<at::Tensor>& offsets,
        const at::Tensor& dense) {
        ctx->save_for_backward(offsets);
        ctx->saved_data["dense_shape"] = dense.sizes();

        auto output = dense.numel() == 0 ? x_values.clone()
                                         : at::empty_like(x_values);

        SYCL_DEVICE_GUARD(dense);

        AT_DISPATCH_SWITCH(
            x_values.scalar_type(),
            "jagged_dense_elementwise_jagged_output_forward",
            AT_DISPATCH_CASE(
                at::ScalarType::Half,
                [&] {
                    jagged_dense_elementwise_jagged_output_opt_<scalar_t>(
                        x_values, offsets, dense, output, JaggedOpAdd());
                })

                FBGEMM_DISPATCH_FLOAT_AND_BFLOAT16_CASE([&] {
                    jagged_dense_elementwise_jagged_output_<scalar_t>(
                        x_values, offsets, dense, output, JaggedOpAdd());
                }));

        return {output};
    }

    static torch::autograd::variable_list backward(
        torch::autograd::AutogradContext* ctx,
        torch::autograd::variable_list grad_outputs) {
        auto offsets = ctx->get_saved_variables();
        auto dense_shape = ctx->saved_data["dense_shape"].toIntVector();
        TORCH_CHECK(grad_outputs.size() == 1);

        SYCL_DEVICE_GUARD(grad_outputs[0]);

        at::Tensor dense_values_grad = jagged_to_padded_dense_forward_xpu(
            grad_outputs[0],
            offsets,
            c10::fromIntArrayRefKnownNonNegative(std::vector<int64_t>(
                dense_shape.begin() + 1, dense_shape.end() - 1)),
            /*padding_value=*/0);
        TORCH_CHECK(dense_values_grad.sizes() == dense_shape);

        return {
            grad_outputs[0],
            torch::autograd::Variable(),  // offsets
            dense_values_grad};
    }
};

}  // namespace

////////////////////////////////////////////////////////////////////////////////
// jagged_dense_elementwise_add_jagged_output_xpu - Host Function
////////////////////////////////////////////////////////////////////////////////
//
// CUDA SOURCE MAPPING:
//   CUDA Function: jagged_dense_elementwise_add_jagged_output_cuda
//   CUDA File: fbgemm_gpu/src/jagged_tensor_ops/jagged_to_padded_dense_forward.cu
//
////////////////////////////////////////////////////////////////////////////////
std::tuple<at::Tensor, std::vector<at::Tensor>>
jagged_dense_elementwise_add_jagged_output_xpu(
    const at::Tensor& x_values,
    const std::vector<at::Tensor>& x_offsets,
    const at::Tensor& y) {
    auto sum_values =
        JaggedDenseAddJaggedOutputXPUOp::apply(x_values, x_offsets, y)[0];

    return {sum_values, x_offsets};
}

/**
 * Register XPU implementation with PyTorch dispatch system.
 *
 * The schema is already declared by the upstream fbgemm-gpu-cpu package
 * (jagged_tensor_ops_cpu.cpp), which fbgemm_xpu/__init__.py imports before _C
 * loads, so only the XPU dispatch key is registered here. Autograd is owned by
 * the upstream Autograd-key wrapper (jagged_tensor_ops_autograd.cpp).
 */
TORCH_LIBRARY_IMPL(fbgemm, XPU, m) {
    m.impl(
        "jagged_dense_elementwise_add_jagged_output",
        &jagged_dense_elementwise_add_jagged_output_xpu);
}

}  // namespace fbgemm_xpu
