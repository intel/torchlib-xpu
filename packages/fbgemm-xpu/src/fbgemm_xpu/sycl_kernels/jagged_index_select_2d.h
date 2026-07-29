/*
 * Copyright (c) Meta Platforms, Inc. and affiliates. All rights reserved.
 * Copyright (c) 2026 Intel Corporation. All Rights Reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */

////////////////////////////////////////////////////////////////////////////////
// SYCL PORT MAPPING TO FBGEMM CUDA SOURCE - JAGGED INDEX SELECT 2D OPERATOR
////////////////////////////////////////////////////////////////////////////////
//
// This file contains SYCL ports of FBGEMM jagged_index_select_2d kernels.
//
// ORIGINAL CUDA SOURCE:
//   File: fbgemm_gpu/src/jagged_tensor_ops/jagged_index_select_2d_forward.cu
//
// KERNEL MAPPING:
//   JaggedIndexSelect2dKernel (SYCL)
//     → jagged_index_select_2d_kernel (CUDA)
//
// HOST FUNCTION MAPPING:
//   jagged_index_select_2d_forward_xpu (SYCL)
//     → jagged_index_select_2d_forward_cuda (CUDA)
//     CUDA File: fbgemm_gpu/src/jagged_tensor_ops/jagged_index_select_2d_forward.cu
//
// DESCRIPTION:
//   Performs 2D index selection on jagged tensors with offset-based indexing.
//   Uses binary search to locate source sequences and block-strided copying
//   for columns.
//
////////////////////////////////////////////////////////////////////////////////

#pragma once

#include <cstdint>

#include <sycl/sycl.hpp>
#include <c10/xpu/XPUStream.h>

#include <ATen/ATen.h>
#include <torch/library.h>

namespace fbgemm_xpu {

////////////////////////////////////////////////////////////////////////////////
// JaggedIndexSelect2dKernel - Device Kernel
////////////////////////////////////////////////////////////////////////////////
//
// CUDA SOURCE MAPPING:
//   CUDA Kernel: jagged_index_select_2d_kernel
//   CUDA File: fbgemm_gpu/src/jagged_tensor_ops/jagged_index_select_2d_forward.cu
//
// DESCRIPTION:
//   Copies sequences from input jagged tensor based on indices specified in
//   the indices tensor to an output jagged tensor. Uses binary search to
//   locate source sequences and block-strided copying for columns.
//
////////////////////////////////////////////////////////////////////////////////
template <typename scalar_t, typename index_t, typename offset_t>
class JaggedIndexSelect2dKernel {
public:
    JaggedIndexSelect2dKernel(
        scalar_t* output,
        const scalar_t* values,
        const index_t* indices,
        const offset_t* input_offsets,
        const offset_t* output_offsets,
        int64_t num_dense_output_rows,
        int64_t num_output_rows,
        int64_t num_cols)
        : output_(output),
          values_(values),
          indices_(indices),
          input_offsets_(input_offsets),
          output_offsets_(output_offsets),
          num_dense_output_rows_(num_dense_output_rows),
          num_output_rows_(num_output_rows),
          num_cols_(num_cols) {}

    void operator()(sycl::nd_item<1> item) const;

private:
    scalar_t* output_;
    const scalar_t* values_;
    const index_t* indices_;
    const offset_t* input_offsets_;
    const offset_t* output_offsets_;
    int64_t num_dense_output_rows_;
    int64_t num_output_rows_;
    int64_t num_cols_;
};

////////////////////////////////////////////////////////////////////////////////
// jagged_index_select_2d_forward_xpu - Host Function
////////////////////////////////////////////////////////////////////////////////
//
// CUDA SOURCE MAPPING:
//   CUDA Function: jagged_index_select_2d_forward_cuda
//   CUDA File: fbgemm_gpu/src/jagged_tensor_ops/jagged_index_select_2d_forward.cu
//
// DESCRIPTION:
//   Host function for dispatching jagged_index_select_2d_kernel to XPU.
//   Validates inputs, allocates output tensor, and launches the kernel with
//   appropriate work-group configuration.
//
// @param values                Input jagged tensor values (2D, on XPU device)
// @param indices               1D tensor of row indices to select
// @param input_offsets         1D tensor of prefix-sum offsets for input rows
// @param output_offsets        1D tensor of prefix-sum offsets for output rows
// @param num_dense_output_rows Total number of dense output rows
// @return at::Tensor Output tensor of shape [num_dense_output_rows, num_cols]
//
////////////////////////////////////////////////////////////////////////////////
at::Tensor jagged_index_select_2d_forward_xpu(
    const at::Tensor& values,
    const at::Tensor& indices,
    const at::Tensor& input_offsets,
    const at::Tensor& output_offsets,
    int64_t num_dense_output_rows);

}  // namespace fbgemm_xpu
