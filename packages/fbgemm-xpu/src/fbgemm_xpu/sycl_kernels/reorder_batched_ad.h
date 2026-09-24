/*
 * Copyright (c) Meta Platforms, Inc. and affiliates. All rights reserved.
 * Copyright (c) 2026 Intel Corporation. All Rights Reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */

////////////////////////////////////////////////////////////////////////////////
// SYCL PORT MAPPING TO FBGEMM CUDA SOURCE - REORDER BATCHED AD OPERATORS
////////////////////////////////////////////////////////////////////////////////
//
// This file contains SYCL ports of FBGEMM batched AD reordering operators.
//
// ORIGINAL CUDA SOURCE:
//   File: fbgemm_gpu/src/sparse_ops/sparse_reorder_batched_ad.cu
//
// KERNEL MAPPING:
//   ReorderBatchedAdLengthsKernel<Dtype>
//     → reorder_batched_ad_lengths_kernel (CUDA)
//
//   NarrowBroadcastIndicesKernel<Dtype, index_t>
//     → narrow_broadcast_indices_kernel (CUDA)
//
//   NarrowBatchedBroadcastIndicesKernel<Dtype, index_t>
//     → narrow_batched_broadcast_indices_kernel (CUDA)
//
//   ReorderBatchedAdIndicesKernel<Dtype, index_t>
//     → reorder_batched_ad_indices_kernel (CUDA)
//
//   ReorderBatchedAdIndicesVecKernel<Dtype, index_t>
//     → reorder_batched_ad_indices_kernel_vec (CUDA)
//
// HOST FUNCTION MAPPING:
//   reorder_batched_ad_lengths_xpu (SYCL)
//     → reorder_batched_ad_lengths_gpu (CUDA)
//     CUDA File: fbgemm_gpu/src/sparse_ops/sparse_reorder_batched_ad.cu
//
//   reorder_batched_ad_indices_xpu (SYCL)
//     → reorder_batched_ad_indices_gpu (CUDA)
//     CUDA File: fbgemm_gpu/src/sparse_ops/sparse_reorder_batched_ad.cu
//
// DESCRIPTION:
//   Reorders batched AD (advertisement) lengths and indices from ragged
//   [B x T x #num_ads_b] layout to [T][B][#num_ads_b] layout for efficient
//   embedding lookups. Supports broadcast modes for lengths and indices.
//
////////////////////////////////////////////////////////////////////////////////

#pragma once

#include <sycl/sycl.hpp>

#include <c10/xpu/XPUStream.h>

#include <ATen/ATen.h>
#include <ATen/DeviceGuard.h>
#include <ATen/xpu/XPUContext.h>
#include <c10/xpu/XPUFunctions.h>
#include <ATen/native/StridedRandomAccessor.h>
#include <torch/library.h>

#include "fbgemm_utils/tensor_utils.h"
#include "fbgemm_utils/utils.h"

using at::native::RestrictPtrTraits;

namespace fbgemm_xpu {

// ============================================================================
// SYCL Kernel Functors
// ============================================================================

////////////////////////////////////////////////////////////////////////////////
// ReorderBatchedAdLengthsKernel - Device Kernel
////////////////////////////////////////////////////////////////////////////////
//
// CUDA SOURCE MAPPING:
//   CUDA Kernel: reorder_batched_ad_lengths_kernel
//   CUDA File: fbgemm_gpu/src/sparse_ops/sparse_reorder_batched_ad.cu
//
// DESCRIPTION:
//   Reorders AD lengths from ragged [B x T x #num_ads_b] to [T][B][#num_ads_b].
//   Each warp processes one (batch, table) pair and copies num_ads_b elements.
//   Supports broadcast mode where a single length is replicated for all ads.
//
////////////////////////////////////////////////////////////////////////////////

/**
 * @brief SYCL kernel functor for reordering AD lengths
 *
 * Warp-per-segment decomposition: each warp handles one (batch, table) pair and
 * copies the `num_ads_b` lengths of that segment to its reordered position.
 *
 * When `broadcast_lengths` is true a single input length is replicated across
 * all ads of the batch.
 */
template <typename Dtype>
class ReorderBatchedAdLengthsKernel {
public:
    ReorderBatchedAdLengthsKernel(
            at::PackedTensorAccessor32<Dtype, 1, RestrictPtrTraits>
                    cat_ad_lengths,
            at::PackedTensorAccessor32<int32_t, 1, RestrictPtrTraits>
                    batch_offsets,
            at::PackedTensorAccessor32<Dtype, 1, RestrictPtrTraits>
                    reordered_cat_ad_lengths,
            int32_t T,
            bool broadcast_lengths)
        : cat_ad_lengths_(cat_ad_lengths),
          batch_offsets_(batch_offsets),
          reordered_cat_ad_lengths_(reordered_cat_ad_lengths),
          T_(T),
          broadcast_lengths_(broadcast_lengths) {}

    void operator()(const sycl::nd_item<2>& item) const;

private:
    at::PackedTensorAccessor32<Dtype, 1, RestrictPtrTraits>
            cat_ad_lengths_;
    at::PackedTensorAccessor32<int32_t, 1, RestrictPtrTraits>
            batch_offsets_;
    // Written by the kernel, so it must stay assignable inside operator() const.
    mutable at::PackedTensorAccessor32<Dtype, 1, RestrictPtrTraits>
            reordered_cat_ad_lengths_;
    int32_t T_;
    bool broadcast_lengths_;
};

////////////////////////////////////////////////////////////////////////////////
// NarrowBroadcastIndicesKernel - Device Kernel (B=1 optimization)
////////////////////////////////////////////////////////////////////////////////
//
// CUDA SOURCE MAPPING:
//   CUDA Kernel: narrow_broadcast_indices_kernel
//   CUDA File: fbgemm_gpu/src/sparse_ops/sparse_reorder_batched_ad.cu
//
// DESCRIPTION:
//   Optimized kernel for the B=1 broadcast case. One warp is assigned to each
//   (table, ad) pair: it reads the single input segment of that table and
//   writes a copy of it into that ad's output slot.
//
////////////////////////////////////////////////////////////////////////////////

/**
 * @brief SYCL kernel functor for the B=1 broadcast fast path
 *
 * One warp per (table, ad) pair. Each warp copies the table's single input
 * segment into one output ad slot, so the segment ends up replicated across all
 * `num_ads_in_batch` slots of that table.
 *
 * Selected only when `broadcast_indices` is true, `T <= 320` and `B == 1`.
 */
template <typename Dtype, typename index_t = int32_t>
class NarrowBroadcastIndicesKernel {
public:
    NarrowBroadcastIndicesKernel(
            at::PackedTensorAccessor32<index_t, 1, RestrictPtrTraits>
                    cat_ad_offsets,
            at::PackedTensorAccessor32<Dtype, 1, RestrictPtrTraits>
                    cat_ad_indices,
            at::PackedTensorAccessor32<Dtype, 1, RestrictPtrTraits>
                    reordered_cat_ad_indices,
            int num_ads_in_batch,
            int reordered_cat_ad_batches)
        : cat_ad_offsets_(cat_ad_offsets),
          cat_ad_indices_(cat_ad_indices),
          reordered_cat_ad_indices_(reordered_cat_ad_indices),
          num_ads_in_batch_(num_ads_in_batch),
          reordered_cat_ad_batches_(reordered_cat_ad_batches) {}

    void operator()(const sycl::nd_item<1>& item) const;

private:
    at::PackedTensorAccessor32<index_t, 1, RestrictPtrTraits>
            cat_ad_offsets_;
    at::PackedTensorAccessor32<Dtype, 1, RestrictPtrTraits>
            cat_ad_indices_;
    // Written by the kernel, so it must stay assignable inside operator() const.
    mutable at::PackedTensorAccessor32<Dtype, 1, RestrictPtrTraits>
            reordered_cat_ad_indices_;
    int num_ads_in_batch_;
    int reordered_cat_ad_batches_;
};

////////////////////////////////////////////////////////////////////////////////
// NarrowBatchedBroadcastIndicesKernel - Device Kernel (B>1 optimization)
////////////////////////////////////////////////////////////////////////////////
//
// CUDA SOURCE MAPPING:
//   CUDA Kernel: narrow_batched_broadcast_indices_kernel
//   CUDA File: fbgemm_gpu/src/sparse_ops/sparse_reorder_batched_ad.cu
//
// DESCRIPTION:
//   Optimized kernel for the 1 < B < 64 broadcast case. The warps of a table
//   are split evenly across the batches (`num_ads_in_batch / B` warps per
//   batch), and those warps stride over the ads of their batch, each copying
//   the batch's single input segment into one ad slot.
//
////////////////////////////////////////////////////////////////////////////////

/**
 * @brief SYCL kernel functor for the 1 < B < 64 broadcast fast path
 *
 * The `num_ads_in_batch` warps of a table are split evenly across the batches,
 * so `num_ads_in_batch / B` warps cooperate on each (table, batch) pair and
 * stride over that batch's ads. Each warp copies the single input segment of
 * its (batch, table) into the ad slot it is responsible for.
 *
 * Selected when `broadcast_indices` is true, `T <= 320` and `1 < B < 64`.
 */
template <typename Dtype, typename index_t = int32_t>
class NarrowBatchedBroadcastIndicesKernel {
public:
    NarrowBatchedBroadcastIndicesKernel(
            at::PackedTensorAccessor32<index_t, 1, RestrictPtrTraits>
                    cat_ad_offsets,
            at::PackedTensorAccessor32<Dtype, 1, RestrictPtrTraits>
                    cat_ad_indices,
            at::PackedTensorAccessor32<index_t, 1, RestrictPtrTraits>
                    reordered_cat_ad_offsets,
            at::PackedTensorAccessor32<Dtype, 1, RestrictPtrTraits>
                    reordered_cat_ad_indices,
            at::PackedTensorAccessor32<int32_t, 1, RestrictPtrTraits>
                    batch_offsets,
            int32_t T)
        : cat_ad_offsets_(cat_ad_offsets),
          cat_ad_indices_(cat_ad_indices),
          reordered_cat_ad_offsets_(reordered_cat_ad_offsets),
          reordered_cat_ad_indices_(reordered_cat_ad_indices),
          batch_offsets_(batch_offsets),
          T_(T) {}

    void operator()(const sycl::nd_item<1>& item) const;

private:
    at::PackedTensorAccessor32<index_t, 1, RestrictPtrTraits>
            cat_ad_offsets_;
    at::PackedTensorAccessor32<Dtype, 1, RestrictPtrTraits>
            cat_ad_indices_;
    at::PackedTensorAccessor32<index_t, 1, RestrictPtrTraits>
            reordered_cat_ad_offsets_;
    // Written by the kernel, so it must stay assignable inside operator() const.
    mutable at::PackedTensorAccessor32<Dtype, 1, RestrictPtrTraits>
            reordered_cat_ad_indices_;
    at::PackedTensorAccessor32<int32_t, 1, RestrictPtrTraits>
            batch_offsets_;
    int32_t T_;
};

////////////////////////////////////////////////////////////////////////////////
// ReorderBatchedAdIndicesKernel - Device Kernel (General case)
////////////////////////////////////////////////////////////////////////////////
//
// CUDA SOURCE MAPPING:
//   CUDA Kernel: reorder_batched_ad_indices_kernel
//   CUDA File: fbgemm_gpu/src/sparse_ops/sparse_reorder_batched_ad.cu
//
// DESCRIPTION:
//   General kernel for reordering indices from [B x T x #num_ads_b x L] to
//   [T][B][#num_ads_b][L]. Each warp processes one (batch, table) pair and
//   copies all indices for all ads in that segment. Handles both broadcast
//   and non-broadcast modes.
//
////////////////////////////////////////////////////////////////////////////////

/**
 * @brief SYCL kernel functor for reordering AD indices (general case)
 *
 * Warp-per-segment decomposition: each warp handles one (batch, table) pair and
 * copies the indices of every ad in that segment. Handles both broadcast and
 * non-broadcast modes, and is the fallback for shapes the narrow kernels above
 * do not cover.
 */
template <typename Dtype, typename index_t = int32_t>
class ReorderBatchedAdIndicesKernel {
public:
    ReorderBatchedAdIndicesKernel(
            at::PackedTensorAccessor32<index_t, 1, RestrictPtrTraits>
                    cat_ad_offsets,
            at::PackedTensorAccessor32<Dtype, 1, RestrictPtrTraits>
                    cat_ad_indices,
            at::PackedTensorAccessor32<index_t, 1, RestrictPtrTraits>
                    reordered_cat_ad_offsets,
            at::PackedTensorAccessor32<Dtype, 1, RestrictPtrTraits>
                    reordered_cat_ad_indices,
            at::PackedTensorAccessor32<int32_t, 1, RestrictPtrTraits>
                    batch_offsets,
            int32_t T,
            bool broadcast_indices)
        : cat_ad_offsets_(cat_ad_offsets),
          cat_ad_indices_(cat_ad_indices),
          reordered_cat_ad_offsets_(reordered_cat_ad_offsets),
          reordered_cat_ad_indices_(reordered_cat_ad_indices),
          batch_offsets_(batch_offsets),
          T_(T),
          broadcast_indices_(broadcast_indices) {}

    void operator()(const sycl::nd_item<2>& item) const;

private:
    at::PackedTensorAccessor32<index_t, 1, RestrictPtrTraits>
            cat_ad_offsets_;
    at::PackedTensorAccessor32<Dtype, 1, RestrictPtrTraits>
            cat_ad_indices_;
    at::PackedTensorAccessor32<index_t, 1, RestrictPtrTraits>
            reordered_cat_ad_offsets_;
    // Written by the kernel, so it must stay assignable inside operator() const.
    mutable at::PackedTensorAccessor32<Dtype, 1, RestrictPtrTraits>
            reordered_cat_ad_indices_;
    at::PackedTensorAccessor32<int32_t, 1, RestrictPtrTraits>
            batch_offsets_;
    int32_t T_;
    bool broadcast_indices_;
};

////////////////////////////////////////////////////////////////////////////////
// ReorderBatchedAdIndicesVecKernel - Device Kernel (General case, vectorized)
////////////////////////////////////////////////////////////////////////////////
//
// CUDA SOURCE MAPPING:
//   CUDA Kernel: reorder_batched_ad_indices_kernel_vec
//   CUDA File: fbgemm_gpu/src/sparse_ops/sparse_reorder_batched_ad.cu
//
// DESCRIPTION:
//   Same decomposition as ReorderBatchedAdIndicesKernel, but the non-broadcast
//   copy uses 2- or 4-wide vector loads/stores for segments longer than 64
//   elements when the dtype is 4 or 8 bytes wide and both pointers carry the
//   required alignment. This is the kernel the CUDA host function launches for
//   the general path; the scalar variant above is kept as the fallback for
//   dtypes the vector path does not cover.
//
////////////////////////////////////////////////////////////////////////////////

/**
 * @brief SYCL kernel functor for reordering AD indices (vectorized general case)
 *
 * Warp-per-segment decomposition, matching ReorderBatchedAdIndicesKernel. The
 * non-broadcast branch copies whole `sycl::vec` units where the segment length
 * and pointer alignment allow, falling back to a scalar copy for misaligned
 * segments and for the trailing elements that do not fill a vector.
 */
template <typename Dtype, typename index_t = int32_t>
class ReorderBatchedAdIndicesVecKernel {
public:
    ReorderBatchedAdIndicesVecKernel(
            at::PackedTensorAccessor32<index_t, 1, RestrictPtrTraits>
                    cat_ad_offsets,
            at::PackedTensorAccessor32<Dtype, 1, RestrictPtrTraits>
                    cat_ad_indices,
            at::PackedTensorAccessor32<index_t, 1, RestrictPtrTraits>
                    reordered_cat_ad_offsets,
            at::PackedTensorAccessor32<Dtype, 1, RestrictPtrTraits>
                    reordered_cat_ad_indices,
            at::PackedTensorAccessor32<int32_t, 1, RestrictPtrTraits>
                    batch_offsets,
            int32_t T,
            bool broadcast_indices)
        : cat_ad_offsets_(cat_ad_offsets),
          cat_ad_indices_(cat_ad_indices),
          reordered_cat_ad_offsets_(reordered_cat_ad_offsets),
          reordered_cat_ad_indices_(reordered_cat_ad_indices),
          batch_offsets_(batch_offsets),
          T_(T),
          broadcast_indices_(broadcast_indices) {}

    void operator()(const sycl::nd_item<2>& item) const;

private:
    at::PackedTensorAccessor32<index_t, 1, RestrictPtrTraits>
            cat_ad_offsets_;
    at::PackedTensorAccessor32<Dtype, 1, RestrictPtrTraits>
            cat_ad_indices_;
    at::PackedTensorAccessor32<index_t, 1, RestrictPtrTraits>
            reordered_cat_ad_offsets_;
    // Written by the kernel, so it must stay assignable inside operator() const.
    mutable at::PackedTensorAccessor32<Dtype, 1, RestrictPtrTraits>
            reordered_cat_ad_indices_;
    at::PackedTensorAccessor32<int32_t, 1, RestrictPtrTraits>
            batch_offsets_;
    int32_t T_;
    bool broadcast_indices_;
};

// ============================================================================
// Host Function Declarations
// ============================================================================

////////////////////////////////////////////////////////////////////////////////
// reorder_batched_ad_lengths_xpu - Host Function
////////////////////////////////////////////////////////////////////////////////
//
// CUDA SOURCE MAPPING:
//   CUDA Function: reorder_batched_ad_lengths_gpu
//   CUDA File: fbgemm_gpu/src/sparse_ops/sparse_reorder_batched_ad.cu
//   CUDA Header: fbgemm_gpu/include/fbgemm_gpu/sparse_ops.h
//
// DESCRIPTION:
//   Host function that reorders the AD lengths tensor to the table-major
//   layout and launches ReorderBatchedAdLengthsKernel.
//
////////////////////////////////////////////////////////////////////////////////

/**
 * @brief XPU implementation of reorder_batched_ad_lengths
 *
 * Reorders AD lengths from [B x T x #num_ads_b] to [T][B][#num_ads_b].
 *
 * @param cat_ad_lengths Input lengths tensor [B x T x #num_ads_b] (ragged)
 * @param batch_offsets Batch offset indices [B+1]
 * @param num_ads_in_batch Total number of ads across all batches
 * @param broadcast_lengths If true, broadcast one length to all ads in a batch
 * @param max_batch_size Unused on XPU; must be <= 0
 * @return Reordered lengths [T x num_ads_in_batch]
 */
at::Tensor reorder_batched_ad_lengths_xpu(
    const at::Tensor& cat_ad_lengths,
    const at::Tensor& batch_offsets,
    const int64_t num_ads_in_batch,
    const bool broadcast_lengths,
    const int64_t max_batch_size);

////////////////////////////////////////////////////////////////////////////////
// reorder_batched_ad_indices_xpu - Host Function
////////////////////////////////////////////////////////////////////////////////
//
// CUDA SOURCE MAPPING:
//   CUDA Function: reorder_batched_ad_indices_gpu
//   CUDA File: fbgemm_gpu/src/sparse_ops/sparse_reorder_batched_ad.cu
//   CUDA Header: fbgemm_gpu/include/fbgemm_gpu/sparse_ops.h
//
// DESCRIPTION:
//   Host function that reorders the AD indices tensor to the table-major
//   layout. Mirrors the CUDA dispatch logic: the narrow broadcast kernels are
//   selected for `broadcast_indices && T <= 320 && B < 64`, otherwise the
//   general ReorderBatchedAdIndicesKernel is launched.
//
////////////////////////////////////////////////////////////////////////////////

/**
 * @brief XPU implementation of reorder_batched_ad_indices
 *
 * Reorders AD indices from [B x T x #num_ads_b x L] to [T][B][#num_ads_b][L].
 *
 * @param cat_ad_offsets Input offset indices [B x T x #num_ads_b + 1] (ragged)
 * @param cat_ad_indices Input indices tensor [sum(L)]
 * @param reordered_cat_ad_offsets Output offset indices [T x num_ads_in_batch + 1]
 * @param batch_offsets Batch offset indices [B+1]
 * @param num_ads_in_batch Total number of ads across all batches
 * @param broadcast_indices If true, broadcast the first ad indices to all ads
 * @param num_indices_after_broadcast Output size when broadcasting; required
 *        to be >= 0 in that case
 * @return Reordered indices
 */
at::Tensor reorder_batched_ad_indices_xpu(
    const at::Tensor& cat_ad_offsets,
    const at::Tensor& cat_ad_indices,
    const at::Tensor& reordered_cat_ad_offsets,
    const at::Tensor& batch_offsets,
    const int64_t num_ads_in_batch,
    const bool broadcast_indices,
    const int64_t num_indices_after_broadcast);

} // namespace fbgemm_xpu
