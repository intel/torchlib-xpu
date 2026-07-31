/*
 * Copyright (c) Meta Platforms, Inc. and affiliates. All rights reserved.
 * Copyright (c) 2026 Intel Corporation. All Rights Reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */

////////////////////////////////////////////////////////////////////////////////
// SYCL PORT MAPPING TO FBGEMM CUDA SOURCE
////////////////////////////////////////////////////////////////////////////////
//
// This file contains SYCL ports of FBGEMM sparse data permutation kernels.
//
// ORIGINAL CUDA SOURCE:
//   File: fbgemm_gpu/src/sparse_ops/sparse_permute_1d.cu
//
// KERNEL MAPPING:
//   Permute1DLengthsKernel (SYCL)
//     → permute_1D_lengths_kernel (CUDA)
//
//   Permute1DDataKernel<has_weight, ...> (SYCL)
//     → permute_1D_data_kernel<has_weight, ...> (CUDA)
//
// HOST FUNCTION MAPPING:
//   permute_1D_sparse_data_xpu (SYCL)
//     → permute_1D_sparse_data_cuda (CUDA)
//     CUDA File: fbgemm_gpu/src/sparse_ops/sparse_permute_1d.cu
//
////////////////////////////////////////////////////////////////////////////////

#pragma once

#include <sycl/sycl.hpp>
#include <c10/xpu/XPUStream.h>

#include <ATen/Operators.h>
#include <torch/all.h>
#include <torch/library.h>


namespace fbgemm_xpu {

// ============================================================================
// SYCL Kernel Functors - Lengths Permutation
// ============================================================================

////////////////////////////////////////////////////////////////////////////////
// Permute1DLengthsKernel - Device Kernel
////////////////////////////////////////////////////////////////////////////////
//
// CUDA SOURCE MAPPING:
//   CUDA Kernel: permute_1D_lengths_kernel
//   CUDA File: fbgemm_gpu/src/sparse_ops/sparse_permute_1d.cu
//
// DESCRIPTION:
//   Simple element-parallel kernel for permuting lengths array according to
//   a permutation index. Each work-item handles one output element:
//   permuted_lengths[i] = lengths[permute[i]]
//
////////////////////////////////////////////////////////////////////////////////

/**
 * @brief SYCL kernel functor for permuting lengths
 *
 * Simple element-parallel kernel: permuted_lengths[i] = lengths[permute[i]]
 */
template <typename index_t, typename permute_t = int32_t>
class Permute1DLengthsKernel {
public:
    Permute1DLengthsKernel(
        int64_t permuted_lengths_size,
        const index_t* lengths,
        const permute_t* permute,
        index_t* permuted_lengths)
        : permuted_lengths_size_(permuted_lengths_size),
          lengths_(lengths),
          permute_(permute),
          permuted_lengths_(permuted_lengths) {}

    void operator()(const sycl::nd_item<1>& item) const;

private:
    int64_t permuted_lengths_size_;
    const index_t* lengths_;
    const permute_t* permute_;
    index_t* permuted_lengths_;
};

// ============================================================================
// SYCL Kernel Functors - Data Permutation
// ============================================================================

////////////////////////////////////////////////////////////////////////////////
// Permute1DDataKernel - Device Kernel
////////////////////////////////////////////////////////////////////////////////
//
// CUDA SOURCE MAPPING:
//   CUDA Kernel: permute_1D_data_kernel<has_weight, offsets_t, indices_t, weights_t>
//   CUDA File: fbgemm_gpu/src/sparse_ops/sparse_permute_1d.cu
//
// DESCRIPTION:
//   Permutes sparse indices (and optionally weights) according to segment
//   permutation. Uses 2D parallel decomposition where each row of work-items
//   processes one segment. Mirrors the CUDA reference by parameterizing the
//   weight-copy path on the non-type template parameter `has_weight`.
//
////////////////////////////////////////////////////////////////////////////////

/**
 * @brief SYCL kernel functor for permuting data (with or without weights)
 *
 * 2D parallel decomposition:
 * - Dimension 0 (y): segments (one row of work-items per segment)
 * - Dimension 1 (x): threads cooperating on one segment
 *
 * When `has_weight` is false, `weights` and `permuted_weights` may be nullptr.
 */
template <
    bool has_weight,
    typename offsets_t,
    typename indices_t,
    typename weights_t>
class Permute1DDataKernel {
public:
    Permute1DDataKernel(
        int64_t permuted_indices_size,
        int64_t permuted_lengths_size,
        const indices_t* indices,
        const weights_t* weights,
        const int32_t* permute,
        const offsets_t* input_offsets,
        const offsets_t* output_offsets,
        indices_t* permuted_indices,
        weights_t* permuted_weights,
        int64_t weight_columns = 1)
        : permuted_indices_size_(permuted_indices_size),
          permuted_lengths_size_(permuted_lengths_size),
          indices_(indices),
          weights_(weights),
          permute_(permute),
          input_offsets_(input_offsets),
          output_offsets_(output_offsets),
          permuted_indices_(permuted_indices),
          permuted_weights_(permuted_weights),
          weight_columns_(weight_columns) {}

    void operator()(const sycl::nd_item<2>& item) const;

private:
    int64_t permuted_indices_size_;
    int64_t permuted_lengths_size_;
    const indices_t* indices_;
    const weights_t* weights_;
    const int32_t* permute_;
    const offsets_t* input_offsets_;
    const offsets_t* output_offsets_;
    indices_t* permuted_indices_;
    weights_t* permuted_weights_;
    int64_t weight_columns_;
};

// ============================================================================
// Host Function Declaration
// ============================================================================

////////////////////////////////////////////////////////////////////////////////
// permute_1D_sparse_data_xpu - Host Function
////////////////////////////////////////////////////////////////////////////////
//
// CUDA SOURCE MAPPING:
//   CUDA Function: permute_1D_sparse_data_cuda
//   CUDA File: fbgemm_gpu/src/sparse_ops/sparse_permute_1d.cu
//   CUDA Header: fbgemm_gpu/include/fbgemm_gpu/sparse_ops.h
//
// DESCRIPTION:
//   Host function that orchestrates the permutation of sparse data in jagged
//   format.
//
////////////////////////////////////////////////////////////////////////////////

/**
 * @brief XPU implementation of permute_1D_sparse_data
 *
 * Permutes sparse data represented in jagged format according to permutation
 * indices.
 */
std::tuple<at::Tensor, at::Tensor, std::optional<at::Tensor>>
permute_1D_sparse_data_xpu(
    const at::Tensor& permute,
    const at::Tensor& lengths,
    const at::Tensor& indices,
    const std::optional<at::Tensor>& weights,
    const std::optional<int64_t>& permuted_lengths_sum);

} // namespace fbgemm_xpu
