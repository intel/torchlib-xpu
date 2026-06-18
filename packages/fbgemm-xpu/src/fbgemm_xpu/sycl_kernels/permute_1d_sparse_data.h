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
//   Permute1DDataKernel (SYCL)
//     → permute_1D_data_kernel<false, ...> (CUDA - no weights variant)
//
//   Permute1DDataWithWeightsKernel (SYCL)
//     → permute_1D_data_kernel<true, ...> (CUDA - with weights variant)
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
template <typename index_t>
class Permute1DLengthsKernel {
public:
    Permute1DLengthsKernel(
        int64_t permuted_lengths_size,
        const index_t* lengths,
        const int32_t* permute,
        index_t* permuted_lengths)
        : permuted_lengths_size_(permuted_lengths_size),
          lengths_(lengths),
          permute_(permute),
          permuted_lengths_(permuted_lengths) {}

    void operator()(const sycl::nd_item<1>& item) const;

private:
    int64_t permuted_lengths_size_;
    const index_t* lengths_;
    const int32_t* permute_;
    index_t* permuted_lengths_;
};

// ============================================================================
// SYCL Kernel Functors - Data Permutation (without weights)
// ============================================================================

////////////////////////////////////////////////////////////////////////////////
// Permute1DDataKernel - Device Kernel
////////////////////////////////////////////////////////////////////////////////
//
// CUDA SOURCE MAPPING:
//   CUDA Kernel: permute_1D_data_kernel<false, offsets_t, indices_t, nullptr_t>
//   CUDA File: fbgemm_gpu/src/sparse_ops/sparse_permute_1d.cu
//
// DESCRIPTION:
//   Permutes sparse indices according to segment permutation. Uses 2D parallel
//   decomposition where each row of work-items processes one segment.
//
////////////////////////////////////////////////////////////////////////////////

/**
 * @brief SYCL kernel functor for permuting data without weights
 *
 * 2D parallel decomposition:
 * - Dimension 0 (y): segments (one row of work-items per segment)
 * - Dimension 1 (x): threads cooperating on one segment
 */
template <typename offsets_t, typename indices_t>
class Permute1DDataKernel {
public:
    Permute1DDataKernel(
        int64_t permuted_indices_size,
        int64_t permuted_lengths_size,
        const indices_t* indices,
        const int32_t* permute,
        const offsets_t* input_offsets,
        const offsets_t* output_offsets,
        indices_t* permuted_indices)
        : permuted_indices_size_(permuted_indices_size),
          permuted_lengths_size_(permuted_lengths_size),
          indices_(indices),
          permute_(permute),
          input_offsets_(input_offsets),
          output_offsets_(output_offsets),
          permuted_indices_(permuted_indices) {}

    void operator()(const sycl::nd_item<2>& item) const;

private:
    int64_t permuted_indices_size_;
    int64_t permuted_lengths_size_;
    const indices_t* indices_;
    const int32_t* permute_;
    const offsets_t* input_offsets_;
    const offsets_t* output_offsets_;
    indices_t* permuted_indices_;
};

// ============================================================================
// SYCL Kernel Functors - Data Permutation (with weights)
// ============================================================================

////////////////////////////////////////////////////////////////////////////////
// Permute1DDataWithWeightsKernel - Device Kernel
////////////////////////////////////////////////////////////////////////////////
//
// CUDA SOURCE MAPPING:
//   CUDA Kernel: permute_1D_data_kernel<true, offsets_t, indices_t, weights_t>
//   CUDA File: fbgemm_gpu/src/sparse_ops/sparse_permute_1d.cu
//
// DESCRIPTION:
//   Same as Permute1DDataKernel but also copies weight values alongside
//   indices.
//
////////////////////////////////////////////////////////////////////////////////

/**
 * @brief SYCL kernel functor for permuting data with weights
 *
 * Same structure as Permute1DDataKernel but also copies weight values.
 */
template <typename offsets_t, typename indices_t, typename weights_t>
class Permute1DDataWithWeightsKernel {
public:
    Permute1DDataWithWeightsKernel(
        int64_t permuted_indices_size,
        int64_t permuted_lengths_size,
        const indices_t* indices,
        const weights_t* weights,
        const int32_t* permute,
        const offsets_t* input_offsets,
        const offsets_t* output_offsets,
        indices_t* permuted_indices,
        weights_t* permuted_weights)
        : permuted_indices_size_(permuted_indices_size),
          permuted_lengths_size_(permuted_lengths_size),
          indices_(indices),
          weights_(weights),
          permute_(permute),
          input_offsets_(input_offsets),
          output_offsets_(output_offsets),
          permuted_indices_(permuted_indices),
          permuted_weights_(permuted_weights) {}

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
