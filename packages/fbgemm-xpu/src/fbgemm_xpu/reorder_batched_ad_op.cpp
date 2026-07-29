/*
 * Copyright (c) Meta Platforms, Inc. and affiliates. All rights reserved.
 * Copyright (c) 2026 Intel Corporation. All Rights Reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */

#include "sycl_kernels/reorder_batched_ad.h"

namespace fbgemm_xpu {

// ============================================================================
// Top-Level Operator Implementations
// ============================================================================

at::Tensor reorder_batched_ad_lengths_xpu(
    const at::Tensor& cat_ad_lengths,
    const at::Tensor& batch_offsets,
    const int64_t num_ads_in_batch,
    const bool broadcast_lengths,
    const int64_t max_batch_size) {
  TORCH_CHECK_LE(max_batch_size, 0);
  TENSORS_ON_SAME_SYCL_XPU_IF_NOT_OPTIONAL(cat_ad_lengths, batch_offsets);

  SYCL_DEVICE_GUARD(cat_ad_lengths);

  const int64_t B = batch_offsets.numel() - 1;
  const int64_t T = broadcast_lengths
      ? cat_ad_lengths.numel() / B
      : cat_ad_lengths.numel() / num_ads_in_batch;

  at::Tensor reordered_cat_ad_lengths = broadcast_lengths
      ? at::empty({T * num_ads_in_batch}, cat_ad_lengths.options())
      : at::empty_like(cat_ad_lengths);

  const int64_t grid_size = (B * T + 32 - 1) / 32;
  TORCH_CHECK(
      grid_size >= 0,
      "grid_size must be positive, got ",
      grid_size,
      " where B =",
      B,
      " and T =",
      T);

  reorder_batched_ad_lengths_kernel_xpu(
      cat_ad_lengths,
      batch_offsets,
      reordered_cat_ad_lengths,
      T,
      broadcast_lengths,
      grid_size);

  return reordered_cat_ad_lengths;
}

at::Tensor reorder_batched_ad_indices_xpu(
    const at::Tensor& cat_ad_offsets,
    const at::Tensor& cat_ad_indices,
    const at::Tensor& reordered_cat_ad_offsets,
    const at::Tensor& batch_offsets,
    const int64_t num_ads_in_batch,
    const bool broadcast_indices,
    const int64_t num_indices_after_broadcast) {
  TENSORS_ON_SAME_SYCL_XPU_IF_NOT_OPTIONAL(
      cat_ad_offsets, cat_ad_indices, reordered_cat_ad_offsets, batch_offsets);

  SYCL_DEVICE_GUARD(cat_ad_offsets);

  const int64_t B = batch_offsets.numel() - 1;
  const int64_t T = (reordered_cat_ad_offsets.numel() - 1) / num_ads_in_batch;
  at::Tensor reordered_cat_ad_indices;
  if (broadcast_indices) {
    TORCH_CHECK_GE(num_indices_after_broadcast, 0);
    reordered_cat_ad_indices =
        at::empty({num_indices_after_broadcast}, cat_ad_indices.options());
  } else {
    reordered_cat_ad_indices = at::empty_like(cat_ad_indices);
  }

  reorder_batched_ad_indices_kernel_xpu(
      cat_ad_offsets,
      cat_ad_indices,
      reordered_cat_ad_offsets,
      batch_offsets,
      reordered_cat_ad_indices,
      num_ads_in_batch,
      B,
      T,
      broadcast_indices);

  return reordered_cat_ad_indices;
}

// ============================================================================
// PyTorch Operator Registration
// ============================================================================

TORCH_LIBRARY_IMPL(fbgemm, XPU, m) {
  m.impl(
      "reorder_batched_ad_lengths",
      &fbgemm_xpu::reorder_batched_ad_lengths_xpu);
  m.impl(
      "reorder_batched_ad_indices",
      &fbgemm_xpu::reorder_batched_ad_indices_xpu);
}

} // namespace fbgemm_xpu
