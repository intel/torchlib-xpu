/*
 * Copyright (c) Meta Platforms, Inc. and affiliates. All rights reserved.
 * Copyright (c) 2026 Intel Corporation. All Rights Reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */

////////////////////////////////////////////////////////////////////////////////
// SYCL PORT MAPPING TO FBGEMM CUDA SOURCE - PERMUTE 2D SPARSE DATA OPERATOR
////////////////////////////////////////////////////////////////////////////////
//
// This file contains SYCL ports of FBGEMM 2D sparse data permutation kernels.
//
// ORIGINAL CUDA SOURCE:
//   File: fbgemm_gpu/src/sparse_ops/sparse_permute_embeddings.cu
//
// KERNEL MAPPING:
//   permute_2D_lengths_kernel_ (SYCL)
//     → permute_2D_lengths_kernel (CUDA)
//
//   permute_2D_data_kernel_ (SYCL)
//     → permute_2D_data_kernel (CUDA)
//
// HOST FUNCTION MAPPING:
//   permute_2D_sparse_data_xpu (SYCL)
//     → permute_2D_sparse_data_cuda (CUDA)
//
// DESCRIPTION:
//   Permutes 2D sparse data (indices, lengths, optional weights) according to
//   a permutation vector. Used for reordering embedding table features.
//
////////////////////////////////////////////////////////////////////////////////

#pragma once

#include <sycl/sycl.hpp>
#include <ATen/ATen.h>
#include <ATen/DeviceGuard.h>
#include <ATen/native/xpu/sycl/KernelUtils.h>
#include <ATen/native/xpu/sycl/ScanUtils.h>
#include <torch/library.h>

#include "../fbgemm_utils/utils.h"
#include "../fbgemm_utils/tensor_utils.h"

// Forward declaration for cumsum kernel
// Implementation is in permute_2d_sparse_data.sycl
namespace at { namespace native { namespace xpu {
  void fbgemm_cumsum_kernel(
      const at::Tensor& result,
      const at::Tensor& self,
      int64_t dim);
}}} // namespace at::native::xpu

namespace fbgemm_xpu {

// ============================================================================
// Helper Functions
// ============================================================================

/**
 * @brief Compute exclusive prefix sum (cumulative sum shifted by one)
 *
 * Computes the exclusive cumulative sum of a 1D int32 or int64 tensor.
 * The output has the same shape as the input, where output[i] = sum(input[0..i-1])
 * and output[0] = 0. Used to convert a lengths tensor into an offsets tensor.
 *
 * @param t_in Input tensor (int32 or int64, contiguous, size < INT_MAX)
 * @return Exclusive cumsum tensor with the same shape as t_in
 */

at::Tensor asynchronous_exclusive_cumsum(const at::Tensor& t_in);

// ============================================================================
// Kernel Functions
// ============================================================================

/**
 * @brief Permute 2D lengths array
 * 
 * Kernel function to permute lengths tensor according to permutation indices.
 * 
 * @param T Number of tables/features
 * @param B Batch size
 * @param lengths_contig Input lengths tensor [T, B]
 * @param permute_contig Permutation indices [T]
 * @param permuted_lengths Output permuted lengths tensor [T, B]
 */
void permute_2D_lengths_kernel_xpu(
    int32_t T,
    int32_t B,
    const at::Tensor& lengths_contig,
    const at::Tensor& permute_contig,
    at::Tensor& permuted_lengths);

/**
 * @brief Permute 2D sparse data (indices and optional weights)
 * 
 * Kernel function to permute sparse indices and weights according to
 * permutation indices and offset information.
 * 
 * @param permuted_indices_size Total size of output indices
 * @param T Number of tables/features
 * @param B Batch size
 * @param indices_contig Input indices
 * @param weights Optional weights tensor
 * @param weights_columns Number of weight columns (for 2D weights)
 * @param permute_contig Permutation indices
 * @param input_offsets Input offset tensor
 * @param output_offsets Output offset tensor
 * @param permuted_indices Output permuted indices
 * @param permuted_weights Optional output permuted weights
 */
void permute_2D_data_kernel_xpu(
    int32_t permuted_indices_size,
    int32_t T,
    int32_t B,
    const at::Tensor& indices_contig,
    const std::optional<const at::Tensor>& weights,
    const int32_t weights_columns,
    const at::Tensor& permute_contig,
    const at::Tensor& input_offsets,
    const at::Tensor& output_offsets,
    at::Tensor& permuted_indices,
    const std::optional<at::Tensor>& permuted_weights);

} // namespace fbgemm_xpu
