/*
 * Copyright (c) Meta Platforms, Inc. and affiliates. All rights reserved.
 * Copyright (c) 2026 Intel Corporation. All Rights Reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */

////////////////////////////////////////////////////////////////////////////////
// SYCL PORT MAPPING TO FBGEMM CUDA SOURCE - JAGGED DENSE ELEMENTWISE MUL
// (BACKWARD)
////////////////////////////////////////////////////////////////////////////////
//
// This file contains the SYCL/XPU kernel and host implementation of the
// jagged_dense_elementwise_mul_backward operator, ported from FBGEMM CUDA
// v1.9.0.
//
// ORIGINAL CUDA SOURCE:
//   File: fbgemm_gpu/src/jagged_tensor_ops/jagged_dense_elementwise_mul_backward.cu
//
// KERNEL MAPPING:
//   JaggedJaggedElementwiseDenseOutputKernel<NUM_JAGGED_DIM, index_t,
//                                            scalar_t, F> (SYCL)
//     → jagged_jagged_elementwise_dense_output_kernel_ (CUDA)
//     CUDA File: fbgemm_gpu/src/jagged_tensor_ops/jagged_dense_elementwise_mul_backward.cu
//
//   The x_values gradient reuses the shared jagged-output launcher in
//   jagged_common.h:
//     JaggedDenseDenseElementwiseJaggedOutputKernel<...>
//       → jagged_dense_dense_elementwise_jagged_output_kernel_ (CUDA)
//     CUDA File: fbgemm_gpu/src/jagged_tensor_ops/common.cuh
//
//   The CUDA `[] __device__ (x, y) { return x * y; }` lambda becomes the
//   JaggedOpMul functor.
//
// HOST FUNCTION MAPPING:
//   jagged_jagged_elementwise_dense_output_ (SYCL)
//     → jagged_jagged_elementwise_dense_output_ (CUDA)
//   jagged_dense_elementwise_mul_backward_xpu (SYCL)
//     → jagged_dense_elementwise_mul_backward (CUDA)
//   CUDA File: fbgemm_gpu/src/jagged_tensor_ops/jagged_dense_elementwise_mul_backward.cu
//
// DESCRIPTION:
//   For output = x * y with x jagged and y dense:
//     x_values_grad = grad_output * y at the positions x occupies (jagged);
//     y_grad        = grad_output * x_values, zero at padded positions (dense).
//   JaggedJaggedElementwiseDenseOutputKernel computes the second: both inputs
//   are jagged with the same offsets and the output is dense.
//
// LAUNCH GEOMETRY:
//   The kernel takes its nd_range from check_shape_and_partition_ in
//   jagged_common.h, like JaggedDenseElementwiseDenseOutputKernel. Dimension 0
//   is CUDA threads.y (dense rows), dimension 1 is CUDA threads.x (inner
//   dense dim), so adjacent work-items touch adjacent elements. That helper
//   sizes the inner extent to ceil(D / 2) and caps the work-group count to
//   DPC++'s int32 flattened-range limit. The kernel therefore walks the outer
//   index with a group-stride loop and the inner dim in element pairs, both
//   matching JaggedDenseElementwiseDenseOutputKernel. CUDA walks one element
//   per thread and caps the grid only for HIP's 2^32 total-thread limit.
//
// XPU DEVIATIONS:
//   (1) An empty y yields a zero-filled x_values_grad. The shared jagged-output
//       launcher returns without writing when y.numel() == 0, so CUDA, which
//       allocates with at::empty_like, returns uninitialized values for a
//       non-empty grad_output. The CPU implementation zero-fills both
//       gradients.
//   (2) x_values must have the shape of grad_output. The y gradient reads
//       x_values at every row and column of grad_output that the offsets
//       cover. CPU and CUDA do not check this, so a smaller x_values reads out
//       of bounds there.
//
// OPERATOR COVERAGE:
//   fbgemm::jagged_dense_elementwise_mul is registered upstream on the
//   Autograd key (JaggedDenseMulOp), whose backward calls this operator.
//   CUDA File: fbgemm_gpu/src/jagged_tensor_ops/jagged_tensor_ops_autograd.cpp
//
////////////////////////////////////////////////////////////////////////////////

#pragma once

#include <tuple>
#include <vector>

#include <ATen/ATen.h>

namespace fbgemm_xpu {

////////////////////////////////////////////////////////////////////////////////
// jagged_dense_elementwise_mul_backward_xpu - Host Function
////////////////////////////////////////////////////////////////////////////////
//
// CUDA SOURCE MAPPING:
//   CUDA Function: jagged_dense_elementwise_mul_backward
//   CUDA File: fbgemm_gpu/src/jagged_tensor_ops/jagged_dense_elementwise_mul_backward.cu
//
// Schema (declared upstream by fbgemm-gpu-cpu, so this repo only supplies the
// XPU dispatch):
//   jagged_dense_elementwise_mul_backward(
//       Tensor grad_output, Tensor[] x_offsets, Tensor y, Tensor x_values)
//       -> (Tensor, Tensor)
//
////////////////////////////////////////////////////////////////////////////////

/**
 * @brief XPU implementation of jagged_dense_elementwise_mul_backward
 *
 * @param grad_output Gradient of the jagged forward output, [total_L, D]
 * @param x_offsets One offsets tensor per jagged dimension; x_offsets[0] has
 *        B+1 entries. At most 5 jagged dimensions are supported.
 * @param y Dense forward operand, [B, max_lengths..., D]
 * @param x_values Jagged forward operand, [total_L, D]
 * @return (x_values_grad [total_L, D], y_grad shaped like y)
 */
std::tuple<at::Tensor, at::Tensor> jagged_dense_elementwise_mul_backward_xpu(
    const at::Tensor& grad_output,
    const std::vector<at::Tensor>& x_offsets,
    const at::Tensor& y,
    const at::Tensor& x_values);

}  // namespace fbgemm_xpu
