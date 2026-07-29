/*
 * Copyright (c) Meta Platforms, Inc. and affiliates. All rights reserved.
 * Copyright (c) 2026 Intel Corporation. All Rights Reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */

////////////////////////////////////////////////////////////////////////////////
// SYCL PORT MAPPING TO FBGEMM CUDA SOURCE - JAGGED TENSOR OPERATORS
////////////////////////////////////////////////////////////////////////////////
//
// This file declares the SYCL/XPU kernel launchers shared by the jagged tensor
// operators dense_to_jagged, jagged_to_padded_dense and
// jagged_dense_elementwise_add_jagged_output.
//
// ORIGINAL CUDA SOURCE:
//   File: fbgemm_gpu/src/jagged_tensor_ops/dense_to_jagged_forward.cu
//   File: fbgemm_gpu/src/jagged_tensor_ops/jagged_to_padded_dense_forward.cu
//   File: fbgemm_gpu/src/jagged_tensor_ops/jagged_dense_dense_elementwise_add_jagged_output_forward.cu
//   Common: fbgemm_gpu/src/jagged_tensor_ops/common.cuh
//
// HOST FUNCTION MAPPING (fbgemm_gpu namespace; no "_cuda" suffix upstream):
//   dense_to_jagged_forward_xpu_kernel (SYCL)
//     -> dense_to_jagged_forward (CUDA, dense_to_jagged_forward.cu)
//   jagged_to_padded_dense_forward_xpu_kernel (SYCL)
//     -> jagged_to_padded_dense_forward (CUDA, jagged_to_padded_dense_forward.cu)
//   jagged_dense_elementwise_add_jagged_output_fwd_xpu_kn (SYCL)
//     -> jagged_dense_dense_elementwise_add_jagged_output_forward
//        (CUDA, jagged_dense_dense_elementwise_add_jagged_output_forward.cu)
//
////////////////////////////////////////////////////////////////////////////////

#pragma once

#include <vector>

#include <sycl/sycl.hpp>

#include <ATen/ATen.h>
#include <ATen/native/xpu/sycl/KernelUtils.h>

#include <comm/SYCLContext.h>

namespace fbgemm_xpu {

// ============================================================================
// Kernel Launchers
// ============================================================================

/**
 * @brief Scatter a dense tensor into jagged (values) layout.
 *
 * Copies the elements of the dense tensor @p y that fall inside the jagged
 * region described by @p x_offsets into @p output_values.
 *
 * @param x_values      Jagged values tensor describing the output layout [N, D]
 * @param x_offsets     Per-dimension prefix-sum offsets defining the jagged rows
 * @param y             Dense source tensor [B, ..., D]
 * @param output_values Destination jagged values tensor [N, D]
 */
void dense_to_jagged_forward_xpu_kernel(
    const at::Tensor& x_values,
    const std::vector<at::Tensor>& x_offsets,
    const at::Tensor& y,
    const at::Tensor& output_values);

/**
 * @brief Expand a jagged tensor into a padded dense tensor.
 *
 * Writes the jagged values @p x_values into the padded dense tensor @p output,
 * filling positions outside the jagged region with @p padding_value.
 *
 * @param x_values      Jagged values tensor [N, D]
 * @param x_offsets     Per-dimension prefix-sum offsets defining the jagged rows
 * @param y             Reference dense tensor providing the output shape
 * @param output        Destination padded dense tensor
 * @param padding_value Value used to fill padded positions
 */
void jagged_to_padded_dense_forward_xpu_kernel(
    const at::Tensor& x_values,
    const std::vector<at::Tensor>& x_offsets,
    const at::Tensor& y,
    const at::Tensor& output,
    const double padding_value = 0.0);

/**
 * @brief Element-wise add of a jagged tensor and a dense tensor (jagged output).
 *
 * Computes output_values = x_values + dense over the jagged region described by
 * @p offsets. Uses a vectorized fast path for half precision when the inputs are
 * suitably aligned, otherwise falls back to the generic per-dimension path.
 *
 * @param x_values      Jagged values tensor [N, D]
 * @param offsets       Per-dimension prefix-sum offsets defining the jagged rows
 * @param dense         Dense operand [B, ..., D]
 * @param output_values Destination jagged values tensor [N, D]
 */
void jagged_dense_elementwise_add_jagged_output_fwd_xpu_kn(
    const at::Tensor& x_values,
    const std::vector<at::Tensor>& offsets,
    const at::Tensor& dense,
    const at::Tensor& output_values);

} // namespace fbgemm_xpu
