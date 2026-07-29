/*
 * Copyright (c) Meta Platforms, Inc. and affiliates. All rights reserved.
 * Copyright (c) 2026 Intel Corporation. All Rights Reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */

#include "asynchronous_complete_cumsum.h"
#include "sycl_kernels/permute_2d_sparse_data.h"

namespace fbgemm_xpu {

// ============================================================================
// Top-Level Operator Function
// ============================================================================

/**
 * @brief Permute 2D sparse data operator
 * 
 * Top-level function that permutes 2D sparse data including lengths, indices,
 * and optional weights according to a permutation vector.
 * 
 * @param permute Permutation indices [T]
 * @param lengths Input lengths [T, B]
 * @param indices Input sparse indices
 * @param weights Optional input weights
 * @param permuted_lengths_sum Optional pre-computed sum of permuted lengths
 * @return Tuple of (permuted_lengths, permuted_indices, permuted_weights)
 */

std::tuple<at::Tensor, at::Tensor, std::optional<at::Tensor>>
permute_2D_sparse_data_xpu(
    const at::Tensor& permute,
    const at::Tensor& lengths,
    const at::Tensor& indices,
    const std::optional<at::Tensor>& weights,
    const std::optional<int64_t>& permuted_lengths_sum) {
  TENSORS_ON_SAME_SYCL_XPU_IF_NOT_OPTIONAL(permute, lengths, indices, weights);
  TORCH_CHECK(lengths.dim() == 2);

  SYCL_DEVICE_GUARD(indices);

  const auto permute_contig = permute.contiguous();
  const auto lengths_contig = lengths.contiguous();
  const auto indices_contig = indices.contiguous();
  // the data to permute over can be less or more with or without
  // repetitions
  const auto T = permute.numel();
  const auto B = lengths.size(1);

  if (T == 0 || B == 0) {
    // When T = 0 or B = 0, permutation will not be performed.  Return the
    // input tensors.
    return {
        lengths.clone(),
        indices.clone(),
        weights.has_value() ? std::make_optional(weights->clone())
                            : std::nullopt};
  }

  at::Tensor permuted_lengths = at::empty({T, B}, lengths.options());
  at::Tensor permuted_indices;
  at::Tensor permuted_weights;

  permute_2D_lengths_kernel_xpu(
      T, B, lengths_contig, permute_contig, permuted_lengths);

  // convert lengths to offsets
  const auto input_offsets = asynchronous_exclusive_cumsum(lengths_contig);
  const auto output_offsets =
      asynchronous_complete_cumsum_xpu(permuted_lengths.flatten());
  int64_t permuted_indices_size = 0;
  if (permuted_lengths_sum.has_value()) {
    permuted_indices_size = permuted_lengths_sum.value();
  } else {
    permuted_indices_size = output_offsets[-1].item<int64_t>();
  }

  permuted_indices = at::empty(permuted_indices_size, indices.options());

  if (weights.has_value()) {
    const at::Tensor weights_value = weights.value();
    int32_t weights_columns = 1;
    if (weights_value.dense_dim() > 1) {
      weights_columns = weights_value.size(1);
      permuted_weights = at::empty(
          {permuted_indices_size, weights_columns}, weights_value.options());
    } else {
      permuted_weights =
          at::empty(permuted_indices_size, weights_value.options());
    }
    permute_2D_data_kernel_xpu(
        permuted_indices_size,
        T,
        B,
        indices_contig,
        std::optional<const at::Tensor>{weights_value},
        weights_columns,
        permute_contig,
        input_offsets,
        output_offsets,
        permuted_indices,
        std::optional<at::Tensor>{permuted_weights});
  } else {
    permute_2D_data_kernel_xpu(
        permuted_indices_size,
        T,
        B,
        indices_contig,
        std::nullopt,
        0,
        permute_contig,
        input_offsets,
        output_offsets,
        permuted_indices,
        std::nullopt);
  }

  return {permuted_lengths, permuted_indices, permuted_weights};
}

TORCH_LIBRARY_IMPL(fbgemm, XPU, m) {
  m.impl("permute_2D_sparse_data", &fbgemm_xpu::permute_2D_sparse_data_xpu);
}

} // namespace fbgemm_xpu
