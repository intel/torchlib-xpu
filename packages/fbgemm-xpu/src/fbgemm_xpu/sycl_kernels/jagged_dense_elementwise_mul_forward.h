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
// This file contains the SYCL/XPU host implementation of the
// jagged_dense_elementwise_mul_forward operator, ported from FBGEMM CUDA
// v1.9.0.
//
// ORIGINAL CUDA SOURCE:
//   File: fbgemm_gpu/src/jagged_tensor_ops/jagged_dense_elementwise_mul_forward.cu
//
// KERNEL MAPPING:
//   None of its own. The operator is expressed entirely through the shared
//   jagged-output launchers in jagged_common.h:
//
//     JaggedDenseDenseElementwiseJaggedOutputKernel<...>
//       → jagged_dense_dense_elementwise_jagged_output_kernel_ (CUDA)
//     JaggedDenseDenseElementwiseJaggedOutputOptSearchKernel<...>
//       → jagged_dense_dense_elementwise_jagged_output_opt_search_kernel_ (CUDA)
//     JaggedDenseDenseElementwiseJaggedOutputOptGatherKernel<...>
//       → jagged_dense_dense_elementwise_jagged_output_opt_gather_kernel_ (CUDA)
//     CUDA File: fbgemm_gpu/src/jagged_tensor_ops/common.cuh
//
//   The CUDA `[] __device__ (x, y) { return x * y; }` lambda becomes the
//   JaggedOpMul functor.
//
// HOST FUNCTION MAPPING:
//   jagged_dense_elementwise_mul_forward_xpu (SYCL)
//     → jagged_dense_elementwise_mul_forward (CUDA)
//     CUDA File: fbgemm_gpu/src/jagged_tensor_ops/jagged_dense_elementwise_mul_forward.cu
//
// DESCRIPTION:
//   Multiplies the jagged values by the dense positions they occupy and returns
//   jagged values with the same layout as x_values. Jagged rows that fall
//   outside the dense tensor (truncation) produce zero.
//
//   Half inputs take the vectorized fast path
//   (jagged_dense_elementwise_jagged_output_opt_); Float and BFloat16 take the
//   generic kernel. This split, and the dtype sets on each side, match CUDA
//   exactly.
//
// XPU DEVIATIONS:
//   (1) An empty dense operand yields a zero-filled output. The shared
//       launchers return without writing when y.numel() == 0, so CUDA, which
//       allocates with at::empty_like, returns uninitialized values for a
//       non-empty x_values. The CPU implementation computes x * 0. This
//       follows the same choice as dense_to_jagged_forward_xpu.
//
// OPERATOR COVERAGE:
//   fbgemm::jagged_dense_elementwise_mul is registered upstream on the
//   Autograd key (JaggedDenseMulOp), whose forward calls this operator.
//   CUDA File: fbgemm_gpu/src/jagged_tensor_ops/jagged_tensor_ops_autograd.cpp
//
////////////////////////////////////////////////////////////////////////////////

#pragma once

#include <vector>

#include <ATen/ATen.h>

namespace fbgemm_xpu {

////////////////////////////////////////////////////////////////////////////////
// jagged_dense_elementwise_mul_forward_xpu - Host Function
////////////////////////////////////////////////////////////////////////////////
//
// CUDA SOURCE MAPPING:
//   CUDA Function: jagged_dense_elementwise_mul_forward
//   CUDA File: fbgemm_gpu/src/jagged_tensor_ops/jagged_dense_elementwise_mul_forward.cu
//
// Schema (declared upstream by fbgemm-gpu-cpu, so this repo only supplies the
// XPU dispatch):
//   jagged_dense_elementwise_mul_forward(
//       Tensor x_values, Tensor[] x_offsets, Tensor y) -> Tensor
//
////////////////////////////////////////////////////////////////////////////////

/**
 * @brief XPU implementation of jagged_dense_elementwise_mul_forward
 *
 * @param x_values Jagged values, [total_L, D]
 * @param x_offsets One offsets tensor per jagged dimension; x_offsets[0] has
 *        B+1 entries. At most 5 jagged dimensions are supported.
 * @param y Dense tensor, [B, max_lengths..., D]
 * @return Jagged values x * y, [total_L, D]
 */
at::Tensor jagged_dense_elementwise_mul_forward_xpu(
    const at::Tensor& x_values,
    const std::vector<at::Tensor>& x_offsets,
    const at::Tensor& y);

}  // namespace fbgemm_xpu
