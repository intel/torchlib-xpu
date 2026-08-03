/*
 * Copyright (c) Meta Platforms, Inc. and affiliates. All rights reserved.
 * Copyright (c) 2026 Intel Corporation. All Rights Reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */

////////////////////////////////////////////////////////////////////////////////
// SYCL PORT MAPPING TO FBGEMM CUDA SOURCE - ASYNCHRONOUS COMPLETE CUMSUM OPERATOR
////////////////////////////////////////////////////////////////////////////////
//
// This file contains SYCL ports of FBGEMM asynchronous_complete_cumsum operator.
//
// ORIGINAL CUDA SOURCE:
//   File: fbgemm_gpu/src/sparse_ops/sparse_async_cumsum.cu
//   Header: fbgemm_gpu/include/fbgemm_gpu/sparse_ops.h
//
// HOST FUNCTION MAPPING:
//   asynchronous_complete_cumsum_xpu (SYCL)
//     → asynchronous_complete_cumsum_gpu (CUDA)
//
// DESCRIPTION:
//   Computes complete cumulative sum with a leading zero on Intel XPU devices.
//   Given input [a, b, c, d], returns [0, a, a+b, a+b+c, a+b+c+d].
//   Supports both 1D and 2D tensors (cumsum along last dimension for 2D).
//   Uses PyTorch's native cumsum operation for computation.
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

} // namespace fbgemm_xpu
