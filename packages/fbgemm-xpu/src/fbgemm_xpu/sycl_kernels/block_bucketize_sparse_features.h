/*
 * Copyright (c) Meta Platforms, Inc. and affiliates. All rights reserved.
 * Copyright (c) 2026 Intel Corporation. All Rights Reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */

/*
 * SYCL/XPU Kernel declarations for block_bucketize_sparse_features
 *
 * Kernel functor classes for block-bucketizing sparse features across ranks,
 * including count, scatter (sequence and pooled), permute, and variable-batch
 * helper kernels.
 */

#pragma once

#include <cstdint>
#include <optional>
#include <vector>

#include <sycl/sycl.hpp>
#include <c10/xpu/XPUStream.h>

#include <ATen/Operators.h>
#include <torch/all.h>
#include <torch/library.h>

namespace fbgemm_xpu {

// Sentinel type used in place of std::nullptr_t when has_weight=false.
// std::nullptr_t is not a valid SYCL kernel name component (SYCL rule:
// kernel names must not use types from reserved namespaces like std::).
struct NoWeightT {};

// ============================================================================
// Kernel 1 – count new_lengths
// One work-item per b_t; serial inner loop over its indices.
// No atomics needed: each work-item is the sole writer to column b_t.
// ============================================================================

template <typename offset_t, typename index_t>
class BlockBucketizeCountKernel {
public:
    BlockBucketizeCountKernel(
        int64_t lengths_size,
        int64_t B,
        int64_t my_size,
        const offset_t* offsets_data,
        const index_t* indices_data,
        const index_t* block_sizes_data,
        const offset_t* length_to_feature_idx,
        const index_t* block_bucketize_pos_concat,
        const index_t* block_bucketize_pos_offsets,
        const index_t* total_num_blocks,
        offset_t* new_lengths_data,
        index_t* indices_to_lb)
      : lengths_size_(lengths_size),
        B_(B),
        my_size_(my_size),
        offsets_data_(offsets_data),
        indices_data_(indices_data),
        block_sizes_data_(block_sizes_data),
        length_to_feature_idx_(length_to_feature_idx),
        block_bucketize_pos_concat_(block_bucketize_pos_concat),
        block_bucketize_pos_offsets_(block_bucketize_pos_offsets),
        total_num_blocks_(total_num_blocks),
        new_lengths_data_(new_lengths_data),
        indices_to_lb_(indices_to_lb) {}

    void operator()(const sycl::nd_item<1>& item) const;

private:
    int64_t lengths_size_, B_, my_size_;
    const offset_t* offsets_data_;
    const index_t* indices_data_;
    const index_t* block_sizes_data_;
    const offset_t* length_to_feature_idx_;
    const index_t* block_bucketize_pos_concat_;
    const index_t* block_bucketize_pos_offsets_;
    const index_t* total_num_blocks_;
    offset_t* new_lengths_data_;
    index_t* indices_to_lb_;
};

// ============================================================================
// Kernel 2 – scatter (sequence path)
// One work-item per b_t; serial inner loop.  Writes unbucketize_permute.
// ============================================================================

template <
    bool has_weight,
    bool bucketize_pos_flag,
    bool return_bucket_mapping,
    typename offset_t,
    typename index_t,
    typename scalar_t>
class BlockBucketizeScatterSeqKernel {
public:
    BlockBucketizeScatterSeqKernel(
        int64_t lengths_size,
        int64_t B,
        int64_t my_size,
        const offset_t* offsets_data,
        const index_t* indices_data,
        const scalar_t* weights_data,
        const index_t* block_sizes_data,
        const offset_t* length_to_feature_idx,
        const index_t* block_bucketize_pos_concat,
        const index_t* block_bucketize_pos_offsets,
        const index_t* indices_to_lb,
        const index_t* total_num_blocks,
        offset_t* new_offsets_data,
        index_t* new_indices_data,
        scalar_t* new_weights_data,
        index_t* new_pos_data,
        index_t* unbucketize_permute_data,
        index_t* bag_mapping_data,
        const bool* keep_orig_idx_per_feature,
        bool keep_orig_idx)
      : lengths_size_(lengths_size), B_(B), my_size_(my_size),
        offsets_data_(offsets_data), indices_data_(indices_data),
        weights_data_(weights_data), block_sizes_data_(block_sizes_data),
        length_to_feature_idx_(length_to_feature_idx),
        block_bucketize_pos_concat_(block_bucketize_pos_concat),
        block_bucketize_pos_offsets_(block_bucketize_pos_offsets),
        indices_to_lb_(indices_to_lb), total_num_blocks_(total_num_blocks),
        new_offsets_data_(new_offsets_data), new_indices_data_(new_indices_data),
        new_weights_data_(new_weights_data), new_pos_data_(new_pos_data),
        unbucketize_permute_data_(unbucketize_permute_data),
        bag_mapping_data_(bag_mapping_data),
        keep_orig_idx_per_feature_(keep_orig_idx_per_feature),
        keep_orig_idx_(keep_orig_idx) {}

    void operator()(const sycl::nd_item<1>& item) const;

private:
    int64_t lengths_size_, B_, my_size_;
    const offset_t* offsets_data_;
    const index_t* indices_data_;
    const scalar_t* weights_data_;
    const index_t* block_sizes_data_;
    const offset_t* length_to_feature_idx_;
    const index_t* block_bucketize_pos_concat_;
    const index_t* block_bucketize_pos_offsets_;
    const index_t* indices_to_lb_;
    const index_t* total_num_blocks_;
    offset_t* new_offsets_data_;
    index_t* new_indices_data_;
    scalar_t* new_weights_data_;
    index_t* new_pos_data_;
    index_t* unbucketize_permute_data_;
    index_t* bag_mapping_data_;
    const bool* keep_orig_idx_per_feature_;
    bool keep_orig_idx_;
};

// ============================================================================
// Kernel 3 – scatter (pooled / non-sequence path)
// One work-item per b_t; serial inner loop.
// No atomics for new_offsets because each work-item owns its column.
// ============================================================================

template <
    bool has_weight,
    bool bucketize_pos_flag,
    typename offset_t,
    typename index_t,
    typename scalar_t>
class BlockBucketizeScatterPooledKernel {
public:
    BlockBucketizeScatterPooledKernel(
        int64_t lengths_size,
        int64_t B,
        int64_t my_size,
        const offset_t* offsets_data,
        const index_t* indices_data,
        const scalar_t* weights_data,
        const index_t* block_sizes_data,
        const offset_t* length_to_feature_idx,
        const index_t* block_bucketize_pos_concat,
        const index_t* block_bucketize_pos_offsets,
        const index_t* indices_to_lb,
        const index_t* total_num_blocks,
        offset_t* new_offsets_data,
        index_t* new_indices_data,
        scalar_t* new_weights_data,
        index_t* new_pos_data,
        const bool* keep_orig_idx_per_feature,
        bool keep_orig_idx)
      : lengths_size_(lengths_size), B_(B), my_size_(my_size),
        offsets_data_(offsets_data), indices_data_(indices_data),
        weights_data_(weights_data), block_sizes_data_(block_sizes_data),
        length_to_feature_idx_(length_to_feature_idx),
        block_bucketize_pos_concat_(block_bucketize_pos_concat),
        block_bucketize_pos_offsets_(block_bucketize_pos_offsets),
        indices_to_lb_(indices_to_lb), total_num_blocks_(total_num_blocks),
        new_offsets_data_(new_offsets_data), new_indices_data_(new_indices_data),
        new_weights_data_(new_weights_data), new_pos_data_(new_pos_data),
        keep_orig_idx_per_feature_(keep_orig_idx_per_feature),
        keep_orig_idx_(keep_orig_idx) {}

    void operator()(const sycl::nd_item<1>& item) const;

private:
    int64_t lengths_size_, B_, my_size_;
    const offset_t* offsets_data_;
    const index_t* indices_data_;
    const scalar_t* weights_data_;
    const index_t* block_sizes_data_;
    const offset_t* length_to_feature_idx_;
    const index_t* block_bucketize_pos_concat_;
    const index_t* block_bucketize_pos_offsets_;
    const index_t* indices_to_lb_;
    const index_t* total_num_blocks_;
    offset_t* new_offsets_data_;
    index_t* new_indices_data_;
    scalar_t* new_weights_data_;
    index_t* new_pos_data_;
    const bool* keep_orig_idx_per_feature_;
    bool keep_orig_idx_;
};

// ============================================================================
// Kernel 4 – populate_bucketized_permute
// One work-item per b_t; serial inner loop.
// ============================================================================

template <typename offset_t, typename index_t>
class PopulateBucketizedPermuteKernel {
public:
    PopulateBucketizedPermuteKernel(
        const offset_t* length_data,
        const offset_t* offset_data,
        offset_t* bucketized_offsets_data,
        const index_t* bucket_mapping_data,
        index_t* bucketized_permute_data,
        int64_t lengths_size)
      : length_data_(length_data), offset_data_(offset_data),
        bucketized_offsets_data_(bucketized_offsets_data),
        bucket_mapping_data_(bucket_mapping_data),
        bucketized_permute_data_(bucketized_permute_data),
        lengths_size_(lengths_size) {}

    void operator()(const sycl::nd_item<1>& item) const;

private:
    const offset_t* length_data_;
    const offset_t* offset_data_;
    offset_t* bucketized_offsets_data_;
    const index_t* bucket_mapping_data_;
    index_t* bucketized_permute_data_;
    int64_t lengths_size_;
};

// ============================================================================
// Helpers: variable-batch length_to_feature_idx population
// ============================================================================

template <typename offset_t>
class PopulateLengthToFeatureIdxKernel {
public:
    PopulateLengthToFeatureIdxKernel(
        int64_t max_B,
        int64_t T,
        const offset_t* batch_size_per_feature,
        const offset_t* batch_size_offsets,
        offset_t* length_to_feature_idx)
      : max_B_(max_B), T_(T),
        batch_size_per_feature_(batch_size_per_feature),
        batch_size_offsets_(batch_size_offsets),
        length_to_feature_idx_(length_to_feature_idx) {}

    void operator()(const sycl::nd_item<1>& item) const;

private:
    int64_t max_B_, T_;
    const offset_t* batch_size_per_feature_;
    const offset_t* batch_size_offsets_;
    offset_t* length_to_feature_idx_;
};

} // namespace fbgemm_xpu
