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
//   File: fbgemm_gpu/src/sparse_ops/sparse_permute_2d.cu
//
// KERNEL MAPPING:
//   Permute2DLengthsKernel<index_t> (SYCL)
//     → permute_2D_lengths_kernel (CUDA)
//
//   Permute2DDataKernel<has_weight, ...> (SYCL)
//     → permute_2D_data_kernel<has_weight, ...> (CUDA)
//
// HOST FUNCTION MAPPING:
//   permute_2D_sparse_data_xpu (SYCL)
//     → permute_2D_sparse_data_cuda (CUDA)
//
//   permute_2D_sparse_preallocated_out_xpu (SYCL)
//     → permute_2D_sparse_preallocated_out_cuda (CUDA)
//
// DESCRIPTION:
//   Permutes 2D sparse data (indices, lengths, optional weights) according to
//   a permutation vector. Used for reordering embedding table features.
//
////////////////////////////////////////////////////////////////////////////////

#pragma once

#include <cstdint>

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
// Permute2DLengthsKernel - Device Kernel
////////////////////////////////////////////////////////////////////////////////
//
// CUDA SOURCE MAPPING:
//   CUDA Kernel: permute_2D_lengths_kernel
//   CUDA File: fbgemm_gpu/src/sparse_ops/sparse_permute_2d.cu
//
// DESCRIPTION:
//   Element-parallel kernel for permuting a lengths tensor of shape [T, B]
//   according to a permutation vector of length T. Each work-item handles one
//   output element:
//     permuted_lengths[t, b] = lengths[permute[t], b]
//   The lengths tensor is treated as a flat buffer of size T * B in row-major
//   layout, so the linear index (t, b) corresponds to storage index t * B + b.
//   A grid-stride loop keeps the kernel scalable for any T * B.
//
////////////////////////////////////////////////////////////////////////////////

/**
 * @brief SYCL kernel functor for permuting 2D lengths.
 *
 * Simple element-parallel kernel:
 *   permuted_lengths[t * B + b] = lengths[permute[t] * B + b]
 */
template <typename index_t>
class Permute2DLengthsKernel {
public:
    Permute2DLengthsKernel(
        int32_t T,
        int32_t B,
        const index_t* lengths,
        const int32_t* permute,
        index_t* permuted_lengths)
        : T_(T),
          B_(B),
          lengths_(lengths),
          permute_(permute),
          permuted_lengths_(permuted_lengths) {}

    void operator()(const sycl::nd_item<1>& item) const;

private:
    int32_t T_;
    int32_t B_;
    const index_t* lengths_;
    const int32_t* permute_;
    index_t* permuted_lengths_;
};

// ============================================================================
// SYCL Kernel Functors - Data Permutation
// ============================================================================

////////////////////////////////////////////////////////////////////////////////
// Permute2DDataKernel - Device Kernel
////////////////////////////////////////////////////////////////////////////////
//
// CUDA SOURCE MAPPING:
//   CUDA Kernel: permute_2D_data_kernel<has_weight, offsets_t, indices_t, weights_t>
//   CUDA File: fbgemm_gpu/src/sparse_ops/sparse_permute_2d.cu
//
// DESCRIPTION:
//   Permutes sparse indices (and optionally weights) according to a segment
//   permutation over T features and B batches. Uses 2D parallel decomposition
//   where each row of work-items processes one (t, b) segment. Mirrors the
//   CUDA reference by parameterizing the weight-copy path on the non-type
//   template parameter `has_weight`, so the no-weights instantiation pays no
//   runtime cost.
//
////////////////////////////////////////////////////////////////////////////////

/**
 * @brief SYCL kernel functor for permuting 2D data (with or without weights).
 *
 * 2D parallel decomposition:
 *   - Dimension 0 (rows): threads cooperating on one segment.
 *   - Dimension 1 (cols): segments (one row of work-items per segment).
 *
 * When `has_weight` is false, `weights` and `permuted_weights` may be nullptr
 * and `weights_columns` is ignored.
 */
template <
    bool has_weight,
    typename offsets_t,
    typename indices_t,
    typename weights_t>
class Permute2DDataKernel {
public:
    Permute2DDataKernel(
        int32_t permuted_indices_size,
        int32_t T,
        int32_t B,
        const indices_t* indices,
        const weights_t* weights,
        int32_t weights_columns,
        const int32_t* permute,
        const offsets_t* input_offsets,
        const offsets_t* output_offsets,
        indices_t* permuted_indices,
        weights_t* permuted_weights)
        : permuted_indices_size_(permuted_indices_size),
          T_(T),
          B_(B),
          indices_(indices),
          weights_(weights),
          weights_columns_(weights_columns),
          permute_(permute),
          input_offsets_(input_offsets),
          output_offsets_(output_offsets),
          permuted_indices_(permuted_indices),
          permuted_weights_(permuted_weights) {}

    void operator()(const sycl::nd_item<2>& item) const;

private:
    int32_t permuted_indices_size_;
    int32_t T_;
    int32_t B_;
    const indices_t* indices_;
    const weights_t* weights_;
    int32_t weights_columns_;
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
// permute_2D_sparse_data_xpu - Host Function
////////////////////////////////////////////////////////////////////////////////
//
// CUDA SOURCE MAPPING:
//   CUDA Function: permute_2D_sparse_data_cuda
//   CUDA File: fbgemm_gpu/src/sparse_ops/sparse_permute_2d.cu
//
// DESCRIPTION:
//   Host function that orchestrates the permutation of 2D sparse data
//   (lengths [T, B], indices, optional weights) according to a permutation
//   vector of length T.
//
////////////////////////////////////////////////////////////////////////////////

/**
 * @brief XPU implementation of permute_2D_sparse_data.
 *
 * Thin wrapper that delegates to permute_2D_sparse_preallocated_out_xpu with
 * no pre-allocated output buffers. Mirrors the FBGEMM CUDA layout where
 * permute_2D_sparse_data_cuda calls permute_2D_sparse_preallocated_out_cuda.
 *
 * @param permute Permutation indices [T] - int32
 * @param lengths Input lengths tensor [T, B] - int32/int64
 * @param indices Concatenated indices tensor - any supported type
 * @param weights Optional weights tensor - floating/half type
 * @param permuted_lengths_sum Optional precomputed sum of permuted lengths
 * @return Tuple of (permuted_lengths, permuted_indices, permuted_weights)
 */
std::tuple<at::Tensor, at::Tensor, std::optional<at::Tensor>>
permute_2D_sparse_data_xpu(
    const at::Tensor& permute,
    const at::Tensor& lengths,
    const at::Tensor& indices,
    const std::optional<at::Tensor>& weights,
    const std::optional<int64_t>& permuted_lengths_sum);

////////////////////////////////////////////////////////////////////////////////
// permute_2D_sparse_preallocated_out_xpu - Host Function
////////////////////////////////////////////////////////////////////////////////
//
// CUDA SOURCE MAPPING:
//   CUDA Function: permute_2D_sparse_preallocated_out_cuda
//   CUDA File: fbgemm_gpu/src/sparse_ops/sparse_permute_2d.cu
//
// DESCRIPTION:
//   Core implementation of the 2D sparse data permutation. Accepts optional
//   pre-allocated output tensors so callers can reuse buffers across
//   invocations. When an output is not provided it is allocated internally.
//
////////////////////////////////////////////////////////////////////////////////

/**
 * @brief XPU core implementation of permute_2D_sparse_preallocated_out.
 *
 * @param permute Permutation indices [T] - int32
 * @param lengths Input lengths tensor [T, B] - int32/int64
 * @param indices Concatenated indices tensor - any supported type
 * @param weights Optional weights tensor - floating/half type
 * @param permuted_lengths_sum Optional precomputed sum of permuted lengths
 * @param permuted_lengths_out Optional pre-allocated permuted_lengths buffer
 * @param permuted_indices_out Optional pre-allocated permuted_indices buffer
 * @param permuted_weights_out Optional pre-allocated permuted_weights buffer
 * @return Tuple of (permuted_lengths, permuted_indices, permuted_weights)
 */
std::tuple<at::Tensor, at::Tensor, std::optional<at::Tensor>>
permute_2D_sparse_preallocated_out_xpu(
    const at::Tensor& permute,
    const at::Tensor& lengths,
    const at::Tensor& indices,
    const std::optional<at::Tensor>& weights,
    const std::optional<int64_t>& permuted_lengths_sum,
    const std::optional<at::Tensor>& permuted_lengths_out,
    const std::optional<at::Tensor>& permuted_indices_out,
    const std::optional<at::Tensor>& permuted_weights_out);

} // namespace fbgemm_xpu
