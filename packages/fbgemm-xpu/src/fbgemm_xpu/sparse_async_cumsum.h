/*
 * Copyright (c) Meta Platforms, Inc. and affiliates. All rights reserved.
 * Copyright (c) 2026 Intel Corporation. All Rights Reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */

////////////////////////////////////////////////////////////////////////////////
// SYCL PORT MAPPING TO FBGEMM CUDA SOURCE - ASYNCHRONOUS CUMSUM OPERATORS
////////////////////////////////////////////////////////////////////////////////
//
// This file contains SYCL ports of FBGEMM asynchronous cumsum operators.
//
// ORIGINAL CUDA SOURCE:
//   File: fbgemm_gpu/src/sparse_ops/sparse_async_cumsum.cu
//   Header: fbgemm_gpu/include/fbgemm_gpu/sparse_ops.h
//
// HOST FUNCTION MAPPING:
//   asynchronous_complete_cumsum_xpu (SYCL)
//     → asynchronous_complete_cumsum_gpu (CUDA)
//   asynchronous_exclusive_cumsum_xpu (SYCL)
//     → asynchronous_exclusive_cumsum_gpu (CUDA)
//   asynchronous_inclusive_cumsum_xpu (SYCL)
//     → asynchronous_inclusive_cumsum_gpu (CUDA)
//
// DESCRIPTION:
//   Computes cumulative sums on Intel XPU devices using PyTorch's native
//   cumsum operation.
//
//   - Complete cumsum: Given input [a, b, c], returns [0, a, a+b, a+b+c].
//     Output has n+1 elements. Supports 1D and 2D tensors.
//
//   - Exclusive cumsum: Given input [a, b, c], returns [0, a, a+b].
//     Output has same size as input. Supports 1D tensors only.
//
//   - Inclusive cumsum: Given input [a, b, c], returns [a, a+b, a+b+c].
//     Output has same size as input. Supports 1D tensors only.
//
////////////////////////////////////////////////////////////////////////////////

#pragma once

#include <ATen/ATen.h>
#include <torch/library.h>

namespace fbgemm_xpu {

/**
 * @brief Computes complete cumulative sum with a leading zero.
 * 
 * This function computes a cumulative sum with a prepended zero element.
 * For a 1D input [a, b, c], it returns [0, a, a+b, a+b+c].
 * For a 2D input, cumsum is computed along dimension 1 (columns).
 * 
 * @param t_in Input tensor (1D or 2D) with dtype int32 or int64
 * @return Output tensor with one additional element along the cumsum dimension
 * 
 * @pre t_in.is_contiguous() == true
 * @pre t_in.dtype() == at::kInt || t_in.dtype() == at::kLong
 * @pre t_in.dim() == 1 || t_in.dim() == 2
 */
at::Tensor asynchronous_complete_cumsum_xpu(const at::Tensor& t_in);

/**
 * @brief Computes exclusive cumulative sum (prefix sum).
 * 
 * This function computes an exclusive prefix sum where each output element
 * is the sum of all preceding input elements (not including itself).
 * For input [a, b, c], it returns [0, a, a+b].
 * 
 * @param t_in Input tensor (1D) with dtype int32 or int64
 * @return Output tensor with same size as input
 * 
 * @pre t_in.is_contiguous() == true
 * @pre t_in.dtype() == at::kInt || t_in.dtype() == at::kLong
 * @pre t_in.dim() == 1
 */
at::Tensor asynchronous_exclusive_cumsum_xpu(const at::Tensor& t_in);

/**
 * @brief Computes inclusive cumulative sum.
 * 
 * This function computes an inclusive cumsum where each output element
 * is the sum of all input elements up to and including itself.
 * For input [a, b, c], it returns [a, a+b, a+b+c].
 * 
 * @param t_in Input tensor (1D) with dtype int32 or int64
 * @return Output tensor with same size as input
 * 
 * @pre t_in.is_contiguous() == true
 * @pre t_in.dtype() == at::kInt || t_in.dtype() == at::kLong
 * @pre t_in.dim() == 1
 */
at::Tensor asynchronous_inclusive_cumsum_xpu(const at::Tensor& t_in);

} // namespace fbgemm_xpu
