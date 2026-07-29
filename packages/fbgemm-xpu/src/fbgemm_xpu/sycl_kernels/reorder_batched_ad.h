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
//   reorder_batched_ad_lengths_kernel_<scalar_t>
//     → reorder_batched_ad_lengths_kernel (CUDA)
//
//   narrow_broadcast_indices_kernel_<scalar_t, index_t>
//     → narrow_broadcast_indices_kernel (CUDA)
//
//   narrow_batched_broadcast_indices_kernel_<scalar_t, index_t>
//     → narrow_batched_broadcast_indices_kernel (CUDA)
//
//   reorder_batched_ad_indices_kernel_<scalar_t, index_t>
//     → reorder_batched_ad_indices_kernel (CUDA)
//
// HOST FUNCTION MAPPING:
//   reorder_batched_ad_lengths_xpu (SYCL)
//     → reorder_batched_ad_lengths_cuda (CUDA)
//
//   reorder_batched_ad_indices_xpu (SYCL)
//     → reorder_batched_ad_indices_cuda (CUDA)
//
// DESCRIPTION:
//   Reorders batched AD (advertisement) lengths and indices from ragged
//   [B x T x #num_ads_b] layout to [T][B][#num_ads_b] layout for efficient
//   embedding lookups. Supports broadcast modes for lengths and indices.
//
////////////////////////////////////////////////////////////////////////////////

#pragma once

#include <sycl/sycl.hpp>
#include <comm/SYCLContext.h>

#include <ATen/ATen.h>
#include <ATen/DeviceGuard.h>
#include <ATen/native/xpu/sycl/KernelUtils.h>
#include <ATen/native/StridedRandomAccessor.h>
#include <torch/library.h>

#include "../fbgemm_utils/utils.h"
#include "../fbgemm_utils/tensor_utils.h"

using at::native::RestrictPtrTraits;

namespace fbgemm_xpu {
// ============================================================================
// Kernel Functions
// ============================================================================

/**
 * @brief Reorder batched AD lengths from [B x T x #num_ads_b] to [T][B][#num_ads_b]
 * 
 * Kernel function to reorder advertisement lengths tensor according to a new layout
 * that groups by table first, then batch. Supports broadcast mode where all ads
 * in a batch share the same length.
 * 
 * @param cat_ad_lengths Input lengths tensor [B x T x #num_ads_b] (ragged)
 * @param batch_offsets Batch offset indices [B+1]
 * @param reordered_cat_ad_lengths Output reordered lengths [T x sum(#num_ads_b)]
 * @param T Number of tables/features
 * @param broadcast_lengths If true, broadcast first length to all ads in batch
 * @param grid_size Number of workgroups for kernel launch
 */
void reorder_batched_ad_lengths_kernel_xpu(
    const at::Tensor& cat_ad_lengths,
    const at::Tensor& batch_offsets,
    at::Tensor& reordered_cat_ad_lengths,
    const int32_t T,
    const bool broadcast_lengths,
    const int32_t grid_size);

/**
 * @brief Reorder batched AD indices from [B x T x #num_ads_b x L] to [T][B][#num_ads_b][L]
 * 
 * Kernel function to reorder advertisement indices tensor according to a new layout
 * that groups by table first, then batch. Supports broadcast mode where indices
 * from first batch are replicated across all batches.
 * 
 * @param cat_ad_offsets Input offset indices [B x T x #num_ads_b + 1] (ragged)
 * @param cat_ad_indices Input indices tensor [sum(L)]
 * @param reordered_cat_ad_offsets Output offset indices [T x sum(#num_ads_b) + 1]
 * @param batch_offsets Batch offset indices [B+1]
 * @param reordered_cat_ad_indices Output reordered indices [sum(L)]
 * @param num_ads_in_batch Total number of ads across all batches
 * @param B Batch size
 * @param T Number of tables/features
 * @param broadcast_indices If true, broadcast first batch indices to all batches
 */
void reorder_batched_ad_indices_kernel_xpu(
    const at::Tensor& cat_ad_offsets,
    const at::Tensor& cat_ad_indices,
    const at::Tensor& reordered_cat_ad_offsets,
    const at::Tensor& batch_offsets,
    at::Tensor& reordered_cat_ad_indices,
    const int64_t num_ads_in_batch,
    const int64_t B,
    const int64_t T,
    const bool broadcast_indices);


} // namespace fbgemm_xpu
