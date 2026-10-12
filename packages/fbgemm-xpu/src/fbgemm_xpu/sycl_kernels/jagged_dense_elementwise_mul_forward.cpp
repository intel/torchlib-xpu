/*
 * Copyright (c) Meta Platforms, Inc. and affiliates. All rights reserved.
 * Copyright (c) 2026 Intel Corporation. All Rights Reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */

////////////////////////////////////////////////////////////////////////////////
// SYCL PORT MAPPING TO FBGEMM CUDA SOURCE - JAGGED DENSE ELEMENTWISE MUL
// (FORWARD)
////////////////////////////////////////////////////////////////////////////////
//
// ORIGINAL CUDA SOURCE:
//   File: fbgemm_gpu/src/jagged_tensor_ops/jagged_dense_elementwise_mul_forward.cu
//   Function: fbgemm_gpu::jagged_dense_elementwise_mul_forward
//
// See jagged_dense_elementwise_mul_forward.h for the full mapping notes.
//
////////////////////////////////////////////////////////////////////////////////

#include <ATen/Dispatch.h>

#include "fbgemm_utils/utils.h"
#include "jagged_common.h"
#include "jagged_dense_elementwise_mul_forward.h"

namespace fbgemm_xpu {

////////////////////////////////////////////////////////////////////////////////
// jagged_dense_elementwise_mul_forward_xpu - Host Function
////////////////////////////////////////////////////////////////////////////////
//
// CUDA SOURCE MAPPING:
//   CUDA Function: jagged_dense_elementwise_mul_forward
//   CUDA File: fbgemm_gpu/src/jagged_tensor_ops/jagged_dense_elementwise_mul_forward.cu
//
////////////////////////////////////////////////////////////////////////////////
at::Tensor jagged_dense_elementwise_mul_forward_xpu(
    const at::Tensor& x_values,
    const std::vector<at::Tensor>& x_offsets,
    const at::Tensor& y) {
    SYCL_DEVICE_GUARD(x_values);

    // Deviation (1): the launchers below write nothing for an empty y.
    at::Tensor output = y.numel() == 0 ? at::zeros_like(x_values)
                                       : at::empty_like(x_values);

    AT_DISPATCH_SWITCH(
        x_values.scalar_type(),
        "jagged_dense_elementwise_mul_jagged_output_forward",
        AT_DISPATCH_CASE(
            at::ScalarType::Half,
            [&] {
                jagged_dense_elementwise_jagged_output_opt_<scalar_t>(
                    x_values, x_offsets, y, output, JaggedOpMul());
            })

            FBGEMM_DISPATCH_FLOAT_AND_BFLOAT16_CASE([&] {
                jagged_dense_elementwise_jagged_output_<scalar_t>(
                    x_values, x_offsets, y, output, JaggedOpMul());
            }));

    return output;
}

/**
 * Register XPU implementation with PyTorch dispatch system.
 *
 * The schema is already declared by the upstream fbgemm-gpu-cpu package
 * (jagged_tensor_ops_cpu.cpp), which fbgemm_xpu/__init__.py imports before _C
 * loads, so only the XPU dispatch key is registered here.
 */
TORCH_LIBRARY_IMPL(fbgemm, XPU, m) {
    m.impl(
        "jagged_dense_elementwise_mul_forward",
        &jagged_dense_elementwise_mul_forward_xpu);
}

}  // namespace fbgemm_xpu
