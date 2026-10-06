/*
 * Copyright (c) Meta Platforms, Inc. and affiliates. All rights reserved.
 * Copyright (c) 2026 Intel Corporation. All Rights Reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */

/*
 * SYCL/XPU Implementation of block_bucketize_sparse_features
 *
 * Original upstream reference (CUDA):
 * https://github.com/pytorch/FBGEMM/blob/main/fbgemm_gpu/src/sparse_ops/sparse_block_bucketize_features.cu
 *
 * This file is a SYCL/XPU adaptation of that implementation for
 * block_bucketize_sparse_features, block_bucketize_sparse_features_inference,
 * and populate_bucketized_permute.
 *
 * Design: two paths with identical results.
 *
 * Serial path: one work-item per (b_t) row for both the count and scatter
 * phases. This avoids per-element atomics on new_lengths/new_offsets
 * because each work-item owns a disjoint column of those arrays, but a
 * row is processed sequentially, so a few long rows (batch-1 inference)
 * leave the device almost idle.
 *
 * Chunked path: rows are split into chunks of kChunkSize indices and each
 * chunk is processed by one sub-group of the same width:
 *   1. BlockBucketizeChunkCountKernel counts each bucket per chunk;
 *   2. a column-wise cumsum over chunks gives every chunk its offset within
 *      its (bucket, row) and BlockBucketizeChunkLengthsKernel derives
 *      new_lengths;
 *   3. BlockBucketizeChunkScatterKernel ranks each index within its chunk
 *      and bucket (sub-group ballot), so positions keep the input order.
 * The output equals the serial path (and the CPU kernel) bit for bit, for
 * pooled rows too. It is chosen for at most kChunkedMaxRows rows with a
 * mean length of at least kChunkedMinMeanRowLength; set
 * FBGEMM_XPU_BLOCK_BUCKETIZE_KERNEL=serial|chunked to force one path.
 *
 * SYCL PORT MAPPING TO FBGEMM CUDA SOURCE
 *   _block_bucketize_sparse_features_cuda_kernel1
 *     -> BlockBucketizeCountKernel (serial), BlockBucketizeChunkCountKernel
 *   _block_bucketize_sequence_sparse_features_cuda_kernel2
 *     -> BlockBucketizeScatterSeqKernel (serial),
 *        BlockBucketizeChunkScatterKernel<sequence=true>
 *   _block_bucketize_pooled_sparse_features_cuda_kernel2
 *     -> BlockBucketizeScatterPooledKernel (serial),
 *        BlockBucketizeChunkScatterKernel<sequence=false>
 *   _populate_bucketized_permute_cuda_kernel -> PopulateBucketizedPermuteKernel
 * Deviations from CUDA:
 *   - CUDA kernel1 counts with atomics and the pooled kernel2 scatters with
 *     shared-memory atomics, so the pooled output order is not defined
 *     there; the CUDA sequence kernel2 is serial per row. The chunked path
 *     instead uses per-chunk counts and a ballot rank (the scheme of
 *     _populate_bucketized_permute_warp_parallel_kernel, generalized to
 *     chunks that run in parallel), which is deterministic and stable.
 *   - The chunked scatter templates only on sequence and the weight type;
 *     bucketize_pos and return_bucket_mapping are runtime nullptr checks.
 *
 * Code-path coverage:
 *   (a) uniform buckets           – block_bucketize_pos == nullptr
 *   (b) variable buckets          – block_bucketize_pos != nullptr, binary search
 *   (c) variable batch per feature – batch_size_per_feature != nullptr
 *       → length_to_feature_idx populated before kernels
 * Both pooled (sequence=false) and sequence (sequence=true) paths,
 * with/without weights, with/without bucketize_pos.
 *
 * unbucketize_permute semantics: unbucketize_permute[original_i] = dest_pos
 * XPU host function parameter order matches ops_registry.cpp schema
 */

#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <optional>
#include <type_traits>
#include <vector>

#include <sycl/sycl.hpp>
#include <c10/xpu/XPUStream.h>

#include <ATen/Operators.h>
#include <torch/all.h>
#include <torch/library.h>

#include "fbgemm_utils/tensor_utils.h"
#include "fbgemm_utils/utils.h"

// FBGEMM dispatch macros (not available in all environments; define inline)
#ifndef FBGEMM_DISPATCH_FLOAT_AND_DOUBLE_CASE
#define FBGEMM_DISPATCH_FLOAT_AND_DOUBLE_CASE(...)     \
  AT_DISPATCH_CASE(at::ScalarType::Float, __VA_ARGS__) \
  AT_DISPATCH_CASE(at::ScalarType::Double, __VA_ARGS__)
#endif
#ifndef FBGEMM_DISPATCH_FLOAT_AND_DOUBLE
#define FBGEMM_DISPATCH_FLOAT_AND_DOUBLE(TYPE, NAME, ...) \
  AT_DISPATCH_SWITCH(                                     \
      TYPE, NAME, FBGEMM_DISPATCH_FLOAT_AND_DOUBLE_CASE(__VA_ARGS__))
#endif

namespace fbgemm_xpu {

// Local complete-cumsum helper for XPU.
// For input [a, b, c] returns [0, a, a+b, a+b+c] preserving the input dtype.
// Avoids a hard dependency on an external asynchronous_complete_cumsum op,
// which is not yet available in this minimal scaffolding.
static at::Tensor local_complete_cumsum_xpu(const at::Tensor& t_in) {
    TORCH_CHECK(t_in.dtype() == at::kInt || t_in.dtype() == at::kLong,
        "block_bucketize_sparse_features: cumsum input must be int32 or int64");
    const int64_t n = t_in.numel();
    auto out = at::zeros({n + 1}, t_in.options());
    if (n > 0) {
        out.slice(0, 1).copy_(t_in.cumsum(0).to(t_in.dtype()));
    }
    return out;
}

// Sentinel type used in place of std::nullptr_t when has_weight=false.
// std::nullptr_t is not a valid SYCL kernel name component (SYCL rule:
// kernel names must not use types from reserved namespaces like std::).
struct NoWeightT {};
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
        const offset_t* length_to_feature_idx, // nullptr if not variable-batch
        const index_t* block_bucketize_pos_concat, // nullptr if uniform
        const index_t* block_bucketize_pos_offsets, // nullptr if uniform
        const index_t* total_num_blocks,            // nullptr if not set
        offset_t* new_lengths_data,
        index_t* indices_to_lb)                     // nullptr if uniform
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

    void operator()(const sycl::nd_item<1>& item) const {
        using uindex_t = std::make_unsigned_t<index_t>;
        const int64_t global_id = item.get_global_id(0);
        const int64_t global_range = item.get_global_range(0);

        for (int64_t b_t = global_id; b_t < lengths_size_; b_t += global_range) {
            const offset_t t = length_to_feature_idx_
                ? static_cast<offset_t>(length_to_feature_idx_[b_t])
                : static_cast<offset_t>(b_t / B_);
            const index_t blk_size = block_sizes_data_[t];

            const index_t local_num_blks = total_num_blocks_
                ? (total_num_blocks_[t] / static_cast<index_t>(my_size_))
                : 1;
            const index_t global_num_blks = total_num_blocks_
                ? total_num_blocks_[t]
                : static_cast<index_t>(my_size_);
            const index_t global_idx_size = blk_size * global_num_blks;
            const index_t local_idx_size  = blk_size * local_num_blks;

            const offset_t rowstart = (b_t == 0) ? 0 : offsets_data_[b_t - 1];
            const offset_t rowend   = offsets_data_[b_t];
            const bool use_bbp = (block_bucketize_pos_concat_ != nullptr);

            if (!use_bbp) {
                // (a) Uniform buckets
                for (offset_t i = rowstart; i < rowend; ++i) {
                    uindex_t idx = static_cast<uindex_t>(indices_data_[i]);
                    uindex_t p = (idx < static_cast<uindex_t>(global_idx_size))
                        ? idx / static_cast<uindex_t>(local_idx_size)
                        : (idx % static_cast<uindex_t>(global_num_blks))
                              / static_cast<uindex_t>(local_num_blks);
                    new_lengths_data_[p * lengths_size_ + b_t]++;
                }
            } else {
                // (b) Variable buckets – binary search
                const index_t first_off = block_bucketize_pos_offsets_[t];
                const index_t last_off  = block_bucketize_pos_offsets_[t + 1];
                // blk_scalar: last boundary / global_num_blks (used when blk_size==0)
                // Use variable-stride index: last_off - 1 is the final boundary entry for feature t
                const uindex_t blk_scalar =
                    (last_off > first_off)
                    ? (static_cast<uindex_t>(block_bucketize_pos_concat_[last_off - 1])
                       / static_cast<uindex_t>(global_num_blks))
                    : static_cast<uindex_t>(1);

                for (offset_t i = rowstart; i < rowend; ++i) {
                    uindex_t idx = static_cast<uindex_t>(indices_data_[i]);
                    if (blk_size == 0) {
                        idx = (idx % static_cast<uindex_t>(global_num_blks)) * blk_scalar;
                    }
                    // Binary search for lower bound
                    index_t lo = first_off, hi = last_off;
                    while (lo < hi) {
                        index_t mid = lo + (hi - lo) / 2;
                        if (static_cast<uindex_t>(block_bucketize_pos_concat_[mid]) <= idx) {
                            lo = mid + 1;
                        } else {
                            hi = mid;
                        }
                    }
                    index_t lb = lo - first_off - 1;
                    if (indices_to_lb_) indices_to_lb_[i] = lb;
                    uindex_t p = (lb < static_cast<index_t>(my_size_))
                        ? static_cast<uindex_t>(lb)
                        : (idx % static_cast<uindex_t>(my_size_));
                    new_lengths_data_[p * lengths_size_ + b_t]++;
                }
            }
        }
    }

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

    void operator()(const sycl::nd_item<1>& item) const {
        using uindex_t = std::make_unsigned_t<index_t>;
        const int64_t global_id = item.get_global_id(0);
        const int64_t global_range = item.get_global_range(0);

        for (int64_t b_t = global_id; b_t < lengths_size_; b_t += global_range) {
            const offset_t t = length_to_feature_idx_
                ? static_cast<offset_t>(length_to_feature_idx_[b_t])
                : static_cast<offset_t>(b_t / B_);
            const index_t blk_size = block_sizes_data_[t];

            const index_t local_num_blks = total_num_blocks_
                ? (total_num_blocks_[t] / static_cast<index_t>(my_size_))
                : 1;
            const index_t global_num_blks = total_num_blocks_
                ? total_num_blocks_[t]
                : static_cast<index_t>(my_size_);
            const index_t global_idx_size = blk_size * global_num_blks;
            const index_t local_idx_size  = blk_size * local_num_blks;

            const offset_t rowstart = (b_t == 0) ? 0 : offsets_data_[b_t - 1];
            const offset_t rowend   = offsets_data_[b_t];
            const bool use_bbp = (block_bucketize_pos_concat_ != nullptr);

            bool keep_idx = keep_orig_idx_;
            if (keep_orig_idx_per_feature_ != nullptr) {
                keep_idx = keep_orig_idx_per_feature_[t];
            }

            for (offset_t i = rowstart; i < rowend; ++i) {
                const uindex_t idx = static_cast<uindex_t>(indices_data_[i]);
                uindex_t p, new_idx;

                if (!use_bbp) {
                    p = (idx < static_cast<uindex_t>(global_idx_size))
                        ? idx / static_cast<uindex_t>(local_idx_size)
                        : (idx % static_cast<uindex_t>(global_num_blks))
                              / static_cast<uindex_t>(local_num_blks);
                    if (keep_idx) {
                        new_idx = idx;
                    } else if (idx < static_cast<uindex_t>(global_idx_size)) {
                        new_idx = idx % static_cast<uindex_t>(local_idx_size);
                    } else {
                        new_idx = idx / static_cast<uindex_t>(global_num_blks);
                    }
                } else {
                    const index_t first_off = block_bucketize_pos_offsets_[t];
                    index_t lb = indices_to_lb_[i];
                    p = (lb < static_cast<index_t>(my_size_))
                        ? static_cast<uindex_t>(lb)
                        : (idx % static_cast<uindex_t>(my_size_));
                    if (keep_idx) {
                        new_idx = idx;
                    } else if (blk_size == 0) {
                        new_idx = idx / static_cast<uindex_t>(global_num_blks);
                    } else if (lb < static_cast<index_t>(my_size_)) {
                        new_idx = idx - static_cast<uindex_t>(
                            block_bucketize_pos_concat_[lb + first_off]);
                    } else {
                        new_idx = idx / static_cast<uindex_t>(my_size_);
                    }
                }

                const offset_t pos = new_offsets_data_[p * lengths_size_ + b_t];
                new_indices_data_[pos] = static_cast<index_t>(new_idx);
                new_offsets_data_[p * lengths_size_ + b_t]++;
                unbucketize_permute_data_[i] = static_cast<index_t>(pos);
                if constexpr (return_bucket_mapping) {
                    bag_mapping_data_[i] = static_cast<index_t>(p);
                }
                if constexpr (has_weight) {
                    new_weights_data_[pos] = weights_data_[i];
                }
                if constexpr (bucketize_pos_flag) {
                    new_pos_data_[pos] = static_cast<index_t>(i - rowstart);
                }
            }
        }
    }

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

    void operator()(const sycl::nd_item<1>& item) const {
        using uindex_t = std::make_unsigned_t<index_t>;
        const int64_t global_id = item.get_global_id(0);
        const int64_t global_range = item.get_global_range(0);

        for (int64_t b_t = global_id; b_t < lengths_size_; b_t += global_range) {
            const offset_t t = length_to_feature_idx_
                ? static_cast<offset_t>(length_to_feature_idx_[b_t])
                : static_cast<offset_t>(b_t / B_);
            const index_t blk_size = block_sizes_data_[t];

            const index_t local_num_blks = total_num_blocks_
                ? (total_num_blocks_[t] / static_cast<index_t>(my_size_))
                : 1;
            const index_t global_num_blks = total_num_blocks_
                ? total_num_blocks_[t]
                : static_cast<index_t>(my_size_);
            const index_t global_idx_size = blk_size * global_num_blks;
            const index_t local_idx_size  = blk_size * local_num_blks;

            const offset_t rowstart = (b_t == 0) ? 0 : offsets_data_[b_t - 1];
            const offset_t rowend   = offsets_data_[b_t];
            const bool use_bbp = (block_bucketize_pos_concat_ != nullptr);

            bool keep_idx = keep_orig_idx_;
            if (keep_orig_idx_per_feature_ != nullptr) {
                keep_idx = keep_orig_idx_per_feature_[t];
            }

            for (offset_t i = rowstart; i < rowend; ++i) {
                const uindex_t idx = static_cast<uindex_t>(indices_data_[i]);
                uindex_t p, new_idx;

                if (!use_bbp) {
                    p = (idx < static_cast<uindex_t>(global_idx_size))
                        ? idx / static_cast<uindex_t>(local_idx_size)
                        : (idx % static_cast<uindex_t>(global_num_blks))
                              / static_cast<uindex_t>(local_num_blks);
                    if (keep_idx) {
                        new_idx = idx;
                    } else if (idx < static_cast<uindex_t>(global_idx_size)) {
                        new_idx = idx % static_cast<uindex_t>(local_idx_size);
                    } else {
                        new_idx = idx / static_cast<uindex_t>(global_num_blks);
                    }
                } else {
                    const index_t first_off = block_bucketize_pos_offsets_[t];
                    index_t lb = indices_to_lb_[i];
                    p = (lb < static_cast<index_t>(my_size_))
                        ? static_cast<uindex_t>(lb)
                        : (idx % static_cast<uindex_t>(my_size_));
                    if (keep_idx) {
                        new_idx = idx;
                    } else if (blk_size == 0) {
                        new_idx = idx / static_cast<uindex_t>(global_num_blks);
                    } else if (lb < static_cast<index_t>(my_size_)) {
                        new_idx = idx - static_cast<uindex_t>(
                            block_bucketize_pos_concat_[lb + first_off]);
                    } else {
                        new_idx = idx / static_cast<uindex_t>(my_size_);
                    }
                }

                // Thread b_t owns column b_t of new_offsets → no atomics
                const offset_t pos = new_offsets_data_[p * lengths_size_ + b_t];
                new_indices_data_[pos] = static_cast<index_t>(new_idx);
                new_offsets_data_[p * lengths_size_ + b_t]++;
                if constexpr (has_weight) {
                    new_weights_data_[pos] = weights_data_[i];
                }
                if constexpr (bucketize_pos_flag) {
                    new_pos_data_[pos] = static_cast<index_t>(i - rowstart);
                }
            }
        }
    }

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

    void operator()(const sycl::nd_item<1>& item) const {
        const int64_t global_id = item.get_global_id(0);
        const int64_t global_range = item.get_global_range(0);

        for (int64_t b_t = global_id; b_t < lengths_size_; b_t += global_range) {
            const offset_t length = length_data_[b_t];
            const offset_t offset = offset_data_[b_t];
            for (offset_t j = 0; j < length; ++j) {
                const offset_t idx = offset + j;
                const index_t bucket = bucket_mapping_data_[idx];
                bucketized_permute_data_[idx] = static_cast<index_t>(
                    bucketized_offsets_data_[bucket * lengths_size_ + b_t]++);
            }
        }
    }

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

    void operator()(const sycl::nd_item<1>& item) const {
        const int64_t b_t = item.get_global_id(0);
        const int64_t t = b_t / max_B_;
        const int64_t b = b_t % max_B_;
        if (t >= T_ || static_cast<offset_t>(b) >= batch_size_per_feature_[t]) return;
        length_to_feature_idx_[batch_size_offsets_[t] + b] = static_cast<offset_t>(t);
    }

private:
    int64_t max_B_, T_;
    const offset_t* batch_size_per_feature_;
    const offset_t* batch_size_offsets_;
    offset_t* length_to_feature_idx_;
};

// ============================================================================
// Chunked path
// One sub-group per chunk of kChunkSize indices of one row; see the file
// comment. The bucket arithmetic below repeats the serial kernels exactly:
// block_bucketize_count_bucket matches BlockBucketizeCountKernel and
// block_bucketize_scatter_bucket matches the two serial scatter kernels.
// ============================================================================

static constexpr int64_t kChunkSize = fbgemm_xpu::kThreadGroupSize;

template <typename index_t>
struct BlockBucketizeFeature {
    index_t blk_size;
    index_t local_num_blks;
    index_t global_num_blks;
    index_t global_idx_size;
    index_t local_idx_size;
};

template <typename offset_t, typename index_t>
inline BlockBucketizeFeature<index_t> block_bucketize_feature(
        offset_t t,
        int64_t my_size,
        const index_t* block_sizes_data,
        const index_t* total_num_blocks) {
    BlockBucketizeFeature<index_t> f;
    f.blk_size = block_sizes_data[t];
    f.local_num_blks = total_num_blocks
        ? (total_num_blocks[t] / static_cast<index_t>(my_size))
        : 1;
    f.global_num_blks = total_num_blocks
        ? total_num_blocks[t]
        : static_cast<index_t>(my_size);
    f.global_idx_size = f.blk_size * f.global_num_blks;
    f.local_idx_size  = f.blk_size * f.local_num_blks;
    return f;
}

// Bucket used for new_lengths; stores the lower bound for variable buckets.
template <typename offset_t, typename index_t>
inline std::make_unsigned_t<index_t> block_bucketize_count_bucket(
        index_t raw_idx,
        offset_t t,
        int64_t my_size,
        const BlockBucketizeFeature<index_t>& f,
        const index_t* block_bucketize_pos_concat,
        const index_t* block_bucketize_pos_offsets,
        index_t* lb_out) {
    using uindex_t = std::make_unsigned_t<index_t>;
    uindex_t idx = static_cast<uindex_t>(raw_idx);
    if (block_bucketize_pos_concat == nullptr) {
        return (idx < static_cast<uindex_t>(f.global_idx_size))
            ? idx / static_cast<uindex_t>(f.local_idx_size)
            : (idx % static_cast<uindex_t>(f.global_num_blks))
                  / static_cast<uindex_t>(f.local_num_blks);
    }
    const index_t first_off = block_bucketize_pos_offsets[t];
    const index_t last_off  = block_bucketize_pos_offsets[t + 1];
    const uindex_t blk_scalar =
        (last_off > first_off)
        ? (static_cast<uindex_t>(block_bucketize_pos_concat[last_off - 1])
           / static_cast<uindex_t>(f.global_num_blks))
        : static_cast<uindex_t>(1);
    if (f.blk_size == 0) {
        idx = (idx % static_cast<uindex_t>(f.global_num_blks)) * blk_scalar;
    }
    index_t lo = first_off, hi = last_off;
    while (lo < hi) {
        index_t mid = lo + (hi - lo) / 2;
        if (static_cast<uindex_t>(block_bucketize_pos_concat[mid]) <= idx) {
            lo = mid + 1;
        } else {
            hi = mid;
        }
    }
    const index_t lb = lo - first_off - 1;
    *lb_out = lb;
    return (lb < static_cast<index_t>(my_size))
        ? static_cast<uindex_t>(lb)
        : (idx % static_cast<uindex_t>(my_size));
}

// Bucket and new index used for the scatter.
template <typename offset_t, typename index_t>
inline std::make_unsigned_t<index_t> block_bucketize_scatter_bucket(
        index_t raw_idx,
        offset_t t,
        int64_t my_size,
        const BlockBucketizeFeature<index_t>& f,
        bool keep_idx,
        const index_t* block_bucketize_pos_concat,
        const index_t* block_bucketize_pos_offsets,
        index_t lb,
        std::make_unsigned_t<index_t>& new_idx) {
    using uindex_t = std::make_unsigned_t<index_t>;
    const uindex_t idx = static_cast<uindex_t>(raw_idx);
    uindex_t p;
    if (block_bucketize_pos_concat == nullptr) {
        p = (idx < static_cast<uindex_t>(f.global_idx_size))
            ? idx / static_cast<uindex_t>(f.local_idx_size)
            : (idx % static_cast<uindex_t>(f.global_num_blks))
                  / static_cast<uindex_t>(f.local_num_blks);
        if (keep_idx) {
            new_idx = idx;
        } else if (idx < static_cast<uindex_t>(f.global_idx_size)) {
            new_idx = idx % static_cast<uindex_t>(f.local_idx_size);
        } else {
            new_idx = idx / static_cast<uindex_t>(f.global_num_blks);
        }
    } else {
        const index_t first_off = block_bucketize_pos_offsets[t];
        p = (lb < static_cast<index_t>(my_size))
            ? static_cast<uindex_t>(lb)
            : (idx % static_cast<uindex_t>(my_size));
        if (keep_idx) {
            new_idx = idx;
        } else if (f.blk_size == 0) {
            new_idx = idx / static_cast<uindex_t>(f.global_num_blks);
        } else if (lb < static_cast<index_t>(my_size)) {
            new_idx = idx - static_cast<uindex_t>(
                block_bucketize_pos_concat[lb + first_off]);
        } else {
            new_idx = idx / static_cast<uindex_t>(my_size);
        }
    }
    return p;
}

inline uint32_t chunk_ballot(const sycl::sub_group& sg, bool predicate) {
    uint32_t bits = 0;
    sycl::ext::oneapi::group_ballot(sg, predicate).extract_bits(bits);
    return bits;
}

// Row b_t that owns chunk: the last row whose first chunk is <= chunk.
// Requires chunk < chunk_offsets[lengths_size].
template <typename offset_t>
inline int64_t chunk_row(
        const offset_t* chunk_offsets, int64_t lengths_size, int64_t chunk) {
    int64_t lo = 0, hi = lengths_size - 1;
    while (lo < hi) {
        const int64_t mid = lo + (hi - lo + 1) / 2;
        if (static_cast<int64_t>(chunk_offsets[mid]) <= chunk) {
            lo = mid;
        } else {
            hi = mid - 1;
        }
    }
    return lo;
}

// Kernel C1 – per-chunk bucket counts: chunk_counts[chunk * my_size + p].
template <typename offset_t, typename index_t>
class BlockBucketizeChunkCountKernel {
public:
    BlockBucketizeChunkCountKernel(
        int64_t lengths_size,
        int64_t B,
        int64_t my_size,
        const offset_t* offsets_data,
        const offset_t* chunk_offsets,
        const index_t* indices_data,
        const index_t* block_sizes_data,
        const offset_t* length_to_feature_idx,
        const index_t* block_bucketize_pos_concat,
        const index_t* block_bucketize_pos_offsets,
        const index_t* total_num_blocks,
        int64_t* chunk_counts,
        index_t* indices_to_lb)
      : lengths_size_(lengths_size), B_(B), my_size_(my_size),
        offsets_data_(offsets_data), chunk_offsets_(chunk_offsets),
        indices_data_(indices_data), block_sizes_data_(block_sizes_data),
        length_to_feature_idx_(length_to_feature_idx),
        block_bucketize_pos_concat_(block_bucketize_pos_concat),
        block_bucketize_pos_offsets_(block_bucketize_pos_offsets),
        total_num_blocks_(total_num_blocks),
        chunk_counts_(chunk_counts), indices_to_lb_(indices_to_lb) {}

    [[sycl::reqd_sub_group_size(kChunkSize)]]
    void operator()(const sycl::nd_item<1>& item) const {
        using uindex_t = std::make_unsigned_t<index_t>;
        const auto sg = item.get_sub_group();
        const int64_t lane = sg.get_local_linear_id();
        const int64_t sg_per_group = sg.get_group_linear_range();
        const int64_t num_chunks = chunk_offsets_[lengths_size_];
        const int64_t stride = item.get_group_range(0) * sg_per_group;

        for (int64_t chunk = item.get_group(0) * sg_per_group + sg.get_group_linear_id();
             chunk < num_chunks;
             chunk += stride) {
            const int64_t b_t = chunk_row(chunk_offsets_, lengths_size_, chunk);
            const offset_t t = length_to_feature_idx_
                ? static_cast<offset_t>(length_to_feature_idx_[b_t])
                : static_cast<offset_t>(b_t / B_);
            const int64_t i = static_cast<int64_t>(offsets_data_[b_t])
                + (chunk - static_cast<int64_t>(chunk_offsets_[b_t])) * kChunkSize + lane;
            const bool valid = i < static_cast<int64_t>(offsets_data_[b_t + 1]);

            uindex_t p = 0;
            if (valid) {
                const auto f = block_bucketize_feature(
                    t, my_size_, block_sizes_data_, total_num_blocks_);
                index_t lb = 0;
                p = block_bucketize_count_bucket(
                    indices_data_[i], t, my_size_, f,
                    block_bucketize_pos_concat_, block_bucketize_pos_offsets_, &lb);
                if (indices_to_lb_) indices_to_lb_[i] = lb;
            }

            // One pass per distinct bucket in the chunk.
            uint32_t pending = chunk_ballot(sg, valid);
            while (pending != 0) {
                const size_t leader = sycl::ctz(pending);
                const uindex_t q = sycl::select_from_group(sg, p, leader);
                const uint32_t same = chunk_ballot(sg, valid && p == q);
                if (static_cast<size_t>(lane) == leader) {
                    chunk_counts_[chunk * my_size_ + static_cast<int64_t>(q)] =
                        sycl::popcount(same);
                }
                pending &= ~same;
            }
        }
    }

private:
    int64_t lengths_size_, B_, my_size_;
    const offset_t* offsets_data_;
    const offset_t* chunk_offsets_;
    const index_t* indices_data_;
    const index_t* block_sizes_data_;
    const offset_t* length_to_feature_idx_;
    const index_t* block_bucketize_pos_concat_;
    const index_t* block_bucketize_pos_offsets_;
    const index_t* total_num_blocks_;
    int64_t* chunk_counts_;
    index_t* indices_to_lb_;
};

// Kernel C2 – new_lengths[p * lengths_size + b_t] from the inclusive
// column-wise cumsum of chunk_counts over chunks.
template <typename offset_t>
class BlockBucketizeChunkLengthsKernel {
public:
    BlockBucketizeChunkLengthsKernel(
        int64_t lengths_size,
        int64_t my_size,
        const offset_t* chunk_offsets,
        const int64_t* chunk_counts_cumsum,
        offset_t* new_lengths_data)
      : lengths_size_(lengths_size), my_size_(my_size),
        chunk_offsets_(chunk_offsets),
        chunk_counts_cumsum_(chunk_counts_cumsum),
        new_lengths_data_(new_lengths_data) {}

    void operator()(const sycl::nd_item<1>& item) const {
        const int64_t total = lengths_size_ * my_size_;
        for (int64_t n = item.get_global_id(0); n < total; n += item.get_global_range(0)) {
            const int64_t p = n / lengths_size_;
            const int64_t b_t = n % lengths_size_;
            const int64_t first = chunk_offsets_[b_t];
            const int64_t end = chunk_offsets_[b_t + 1];
            int64_t count = 0;
            if (end > first) {
                count = chunk_counts_cumsum_[(end - 1) * my_size_ + p]
                    - (first > 0 ? chunk_counts_cumsum_[(first - 1) * my_size_ + p] : 0);
            }
            new_lengths_data_[n] = static_cast<offset_t>(count);
        }
    }

private:
    int64_t lengths_size_, my_size_;
    const offset_t* chunk_offsets_;
    const int64_t* chunk_counts_cumsum_;
    offset_t* new_lengths_data_;
};

// Kernel C3 – scatter. An index lands at
//   new_offsets[p * lengths_size + b_t]       (start of bucket p in row b_t)
//   + earlier chunks of the row in bucket p   (from the cumsum)
//   + earlier lanes of the chunk in bucket p  (ballot rank),
// which is the position the serial kernels assign.
template <bool sequence, typename offset_t, typename index_t, typename scalar_t>
class BlockBucketizeChunkScatterKernel {
public:
    static constexpr bool has_weight = !std::is_same_v<scalar_t, NoWeightT>;

    BlockBucketizeChunkScatterKernel(
        int64_t lengths_size,
        int64_t B,
        int64_t my_size,
        const offset_t* offsets_data,
        const offset_t* chunk_offsets,
        const int64_t* chunk_counts_cumsum,
        const index_t* indices_data,
        const scalar_t* weights_data,
        const index_t* block_sizes_data,
        const offset_t* length_to_feature_idx,
        const index_t* block_bucketize_pos_concat,
        const index_t* block_bucketize_pos_offsets,
        const index_t* indices_to_lb,
        const index_t* total_num_blocks,
        const offset_t* new_offsets_data,
        index_t* new_indices_data,
        scalar_t* new_weights_data,
        index_t* new_pos_data,
        index_t* unbucketize_permute_data,
        index_t* bag_mapping_data,
        const bool* keep_orig_idx_per_feature,
        bool keep_orig_idx)
      : lengths_size_(lengths_size), B_(B), my_size_(my_size),
        offsets_data_(offsets_data), chunk_offsets_(chunk_offsets),
        chunk_counts_cumsum_(chunk_counts_cumsum),
        indices_data_(indices_data), weights_data_(weights_data),
        block_sizes_data_(block_sizes_data),
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

    [[sycl::reqd_sub_group_size(kChunkSize)]]
    void operator()(const sycl::nd_item<1>& item) const {
        using uindex_t = std::make_unsigned_t<index_t>;
        const auto sg = item.get_sub_group();
        const int64_t lane = sg.get_local_linear_id();
        const uint32_t lanes_below = (uint32_t{1} << lane) - 1;
        const int64_t sg_per_group = sg.get_group_linear_range();
        const int64_t num_chunks = chunk_offsets_[lengths_size_];
        const int64_t stride = item.get_group_range(0) * sg_per_group;

        for (int64_t chunk = item.get_group(0) * sg_per_group + sg.get_group_linear_id();
             chunk < num_chunks;
             chunk += stride) {
            const int64_t b_t = chunk_row(chunk_offsets_, lengths_size_, chunk);
            const offset_t t = length_to_feature_idx_
                ? static_cast<offset_t>(length_to_feature_idx_[b_t])
                : static_cast<offset_t>(b_t / B_);
            const int64_t first_chunk = chunk_offsets_[b_t];
            const offset_t rowstart = offsets_data_[b_t];
            const int64_t i = static_cast<int64_t>(rowstart)
                + (chunk - first_chunk) * kChunkSize + lane;
            const bool valid = i < static_cast<int64_t>(offsets_data_[b_t + 1]);

            uindex_t p = 0;
            uindex_t new_idx = 0;
            if (valid) {
                const auto f = block_bucketize_feature(
                    t, my_size_, block_sizes_data_, total_num_blocks_);
                const bool keep_idx = keep_orig_idx_per_feature_ != nullptr
                    ? keep_orig_idx_per_feature_[t]
                    : keep_orig_idx_;
                p = block_bucketize_scatter_bucket(
                    indices_data_[i], t, my_size_, f, keep_idx,
                    block_bucketize_pos_concat_, block_bucketize_pos_offsets_,
                    indices_to_lb_ ? indices_to_lb_[i] : index_t{0}, new_idx);
            }

            int64_t rank = 0;
            uint32_t pending = chunk_ballot(sg, valid);
            while (pending != 0) {
                const size_t leader = sycl::ctz(pending);
                const uindex_t q = sycl::select_from_group(sg, p, leader);
                const bool mine = valid && p == q;
                const uint32_t same = chunk_ballot(sg, mine);
                if (mine) rank = sycl::popcount(same & lanes_below);
                pending &= ~same;
            }

            if (valid) {
                const int64_t column = static_cast<int64_t>(p);
                const int64_t earlier_chunks =
                    (chunk > 0 ? chunk_counts_cumsum_[(chunk - 1) * my_size_ + column] : 0)
                    - (first_chunk > 0
                           ? chunk_counts_cumsum_[(first_chunk - 1) * my_size_ + column]
                           : 0);
                const offset_t pos = new_offsets_data_[column * lengths_size_ + b_t]
                    + static_cast<offset_t>(earlier_chunks + rank);
                new_indices_data_[pos] = static_cast<index_t>(new_idx);
                if constexpr (sequence) {
                    unbucketize_permute_data_[i] = static_cast<index_t>(pos);
                    if (bag_mapping_data_) {
                        bag_mapping_data_[i] = static_cast<index_t>(p);
                    }
                }
                if constexpr (has_weight) {
                    new_weights_data_[pos] = weights_data_[i];
                }
                if (new_pos_data_) {
                    new_pos_data_[pos] = static_cast<index_t>(i - rowstart);
                }
            }
        }
    }

private:
    int64_t lengths_size_, B_, my_size_;
    const offset_t* offsets_data_;
    const offset_t* chunk_offsets_;
    const int64_t* chunk_counts_cumsum_;
    const index_t* indices_data_;
    const scalar_t* weights_data_;
    const index_t* block_sizes_data_;
    const offset_t* length_to_feature_idx_;
    const index_t* block_bucketize_pos_concat_;
    const index_t* block_bucketize_pos_offsets_;
    const index_t* indices_to_lb_;
    const index_t* total_num_blocks_;
    const offset_t* new_offsets_data_;
    index_t* new_indices_data_;
    scalar_t* new_weights_data_;
    index_t* new_pos_data_;
    index_t* unbucketize_permute_data_;
    index_t* bag_mapping_data_;
    const bool* keep_orig_idx_per_feature_;
    bool keep_orig_idx_;
};

// ============================================================================
// Host-side launcher helpers
// ============================================================================

static constexpr int64_t kThreads = 256;

static int64_t grid_size(int64_t n) {
    return (n + kThreads - 1) / kThreads;
}

// Exclusive cumsum returning tensor of size N (no +1, C-style exclusive scan)
static at::Tensor xpu_excl_cumsum(const at::Tensor& t) {
    // Build a +1 sized inclusive cumsum and take the first N elements
    at::Tensor inc = fbgemm_xpu::local_complete_cumsum_xpu(t);
    // inc has N+1 elements: [0, t[0], t[0]+t[1], ...]
    // We want exclusive: [0, t[0], t[0]+t[1], ...] (first N elements of inc)
    return inc.slice(0, 0, t.numel());
}

// ============================================================================
// Chunked path: selection and launch
// ============================================================================

// The serial kernels already fill the device with many rows, and short rows
// cost them little. On Data Center GPU Max the chunked path was faster from a
// mean row length of 128 up to 8192 rows, equal near 13k rows and slower at
// 53k rows. Few long rows mixed into many short ones still take the serial
// path.
static constexpr int64_t kChunkedMinMeanRowLength = 128;
static constexpr int64_t kChunkedMaxRows = 8192;
// Limit for the per-chunk count buffer (int64 entries, 128 MiB).
static constexpr int64_t kChunkedMaxCounts = int64_t{1} << 24;

static int64_t chunked_max_chunks(int64_t lengths_sum, int64_t lengths_size) {
    // Each row adds at most one partial chunk.
    return div_round_up(lengths_sum, kChunkSize) + lengths_size;
}

static bool use_chunked_block_bucketize(
        int64_t lengths_sum, int64_t lengths_size, int64_t my_size) {
    const char* choice = std::getenv("FBGEMM_XPU_BLOCK_BUCKETIZE_KERNEL");
    if (choice != nullptr && std::strcmp(choice, "serial") == 0) {
        return false;
    }
    if (choice != nullptr && std::strcmp(choice, "chunked") == 0) {
        return true;
    }
    TORCH_CHECK(
        choice == nullptr || choice[0] == '\0' || std::strcmp(choice, "auto") == 0,
        "FBGEMM_XPU_BLOCK_BUCKETIZE_KERNEL must be auto, serial or chunked, got '",
        choice, "'");
    return lengths_size > 0 && lengths_size <= kChunkedMaxRows && my_size > 0 &&
        lengths_sum >= kChunkedMinMeanRowLength * lengths_size &&
        chunked_max_chunks(lengths_sum, lengths_size) <= kChunkedMaxCounts / my_size;
}

static void block_bucketize_chunked_xpu(
        sycl::queue& queue,
        const at::Tensor& lengths_contig,
        const at::Tensor& indices_contig,
        const at::Tensor& offsets,
        const at::Tensor& block_sizes,
        const std::optional<at::Tensor>& total_num_blocks,
        const int64_t B,
        const int64_t my_size,
        const bool sequence,
        const std::optional<at::Tensor>& weights_contig,
        const std::optional<at::Tensor>& keep_orig_idx_per_feature,
        const bool keep_orig_idx,
        const std::optional<at::Tensor>& length_to_feature_idx,
        const std::optional<at::Tensor>& bbp_concat,
        const std::optional<at::Tensor>& bbp_offsets,
        at::Tensor& indices_to_lb,
        at::Tensor& new_lengths,
        at::Tensor& new_indices,
        std::optional<at::Tensor>& new_weights,
        std::optional<at::Tensor>& new_pos,
        std::optional<at::Tensor>& unbucketize_permute,
        std::optional<at::Tensor>& bucket_mapping) {
    const int64_t lengths_size = lengths_contig.numel();
    if (lengths_size == 0) {
        return;
    }
    const int64_t lengths_sum = indices_contig.numel();
    const int64_t max_chunks = chunked_max_chunks(lengths_sum, lengths_size);
    const int64_t sg_per_group = kThreads / kChunkSize;
    const int64_t chunk_groups = xpu_cap_grid_dim_x(
        div_round_up(max_chunks, sg_per_group), kThreads);
    const int64_t length_groups = xpu_cap_grid_dim_x(
        grid_size(lengths_size * my_size), kThreads);
    const sycl::nd_range<1> chunk_range(
        sycl::range<1>(chunk_groups * kThreads), sycl::range<1>(kThreads));

    const auto chunk_offsets = fbgemm_xpu::local_complete_cumsum_xpu(
        at::div(lengths_contig + (kChunkSize - 1), kChunkSize, "floor"));
    auto chunk_counts = at::zeros(
        {max_chunks, my_size}, lengths_contig.options().dtype(at::kLong));

    AT_DISPATCH_INDEX_TYPES(
        lengths_contig.scalar_type(), "block_bucketize_chunked_xpu_1", [&] {
            using offset_t = index_t;
            AT_DISPATCH_INDEX_TYPES(
                indices_contig.scalar_type(), "block_bucketize_chunked_xpu_2", [&] {
                    const offset_t* length_to_feature_idx_data =
                        length_to_feature_idx.has_value()
                        ? length_to_feature_idx->data_ptr<offset_t>()
                        : nullptr;
                    const index_t* bbp_concat_data =
                        bbp_concat.has_value() ? bbp_concat->data_ptr<index_t>() : nullptr;
                    const index_t* bbp_offsets_data =
                        bbp_offsets.has_value() ? bbp_offsets->data_ptr<index_t>() : nullptr;
                    index_t* indices_to_lb_data =
                        bbp_concat.has_value() ? indices_to_lb.data_ptr<index_t>() : nullptr;
                    const index_t* total_num_blocks_data = total_num_blocks.has_value()
                        ? total_num_blocks->data_ptr<index_t>()
                        : nullptr;

                    queue.submit([&](sycl::handler& cgh) {
                        cgh.parallel_for<BlockBucketizeChunkCountKernel<offset_t, index_t>>(
                            chunk_range,
                            BlockBucketizeChunkCountKernel<offset_t, index_t>(
                                lengths_size, B, my_size,
                                offsets.data_ptr<offset_t>(),
                                chunk_offsets.data_ptr<offset_t>(),
                                indices_contig.data_ptr<index_t>(),
                                block_sizes.data_ptr<index_t>(),
                                length_to_feature_idx_data,
                                bbp_concat_data,
                                bbp_offsets_data,
                                total_num_blocks_data,
                                chunk_counts.data_ptr<int64_t>(),
                                indices_to_lb_data));
                    });

                    const auto chunk_counts_cumsum = chunk_counts.cumsum(0);

                    queue.submit([&](sycl::handler& cgh) {
                        cgh.parallel_for<BlockBucketizeChunkLengthsKernel<offset_t>>(
                            sycl::nd_range<1>(
                                sycl::range<1>(length_groups * kThreads),
                                sycl::range<1>(kThreads)),
                            BlockBucketizeChunkLengthsKernel<offset_t>(
                                lengths_size, my_size,
                                chunk_offsets.data_ptr<offset_t>(),
                                chunk_counts_cumsum.data_ptr<int64_t>(),
                                new_lengths.data_ptr<offset_t>()));
                    });

                    const auto new_offsets = xpu_excl_cumsum(new_lengths);

                    auto scatter = [&]<bool kSequence, typename scalar_t>(
                            const scalar_t* weights_data, scalar_t* new_weights_data) {
                        queue.submit([&](sycl::handler& cgh) {
                            cgh.parallel_for<BlockBucketizeChunkScatterKernel<
                                kSequence, offset_t, index_t, scalar_t>>(
                                chunk_range,
                                BlockBucketizeChunkScatterKernel<
                                    kSequence, offset_t, index_t, scalar_t>(
                                    lengths_size, B, my_size,
                                    offsets.data_ptr<offset_t>(),
                                    chunk_offsets.data_ptr<offset_t>(),
                                    chunk_counts_cumsum.data_ptr<int64_t>(),
                                    indices_contig.data_ptr<index_t>(),
                                    weights_data,
                                    block_sizes.data_ptr<index_t>(),
                                    length_to_feature_idx_data,
                                    bbp_concat_data,
                                    bbp_offsets_data,
                                    indices_to_lb_data,
                                    total_num_blocks_data,
                                    new_offsets.data_ptr<offset_t>(),
                                    new_indices.data_ptr<index_t>(),
                                    new_weights_data,
                                    new_pos.has_value() ? new_pos->data_ptr<index_t>() : nullptr,
                                    kSequence ? unbucketize_permute->data_ptr<index_t>() : nullptr,
                                    kSequence && bucket_mapping.has_value()
                                        ? bucket_mapping->data_ptr<index_t>()
                                        : nullptr,
                                    keep_orig_idx_per_feature.has_value()
                                        ? keep_orig_idx_per_feature->const_data_ptr<bool>()
                                        : nullptr,
                                    keep_orig_idx));
                        });
                    };

                    if (weights_contig.has_value()) {
                        FBGEMM_DISPATCH_FLOAT_AND_DOUBLE(
                            weights_contig->scalar_type(), "block_bucketize_chunked_xpu_3", [&] {
                                const scalar_t* weights_data = weights_contig->data_ptr<scalar_t>();
                                scalar_t* new_weights_data = new_weights->data_ptr<scalar_t>();
                                if (sequence) {
                                    scatter.template operator()<true, scalar_t>(
                                        weights_data, new_weights_data);
                                } else {
                                    scatter.template operator()<false, scalar_t>(
                                        weights_data, new_weights_data);
                                }
                            });
                    } else if (sequence) {
                        scatter.template operator()<true, NoWeightT>(nullptr, nullptr);
                    } else {
                        scatter.template operator()<false, NoWeightT>(nullptr, nullptr);
                    }
                });
        });
}

// ============================================================================
// Core XPU implementation
// ============================================================================

static std::tuple<
    at::Tensor,
    at::Tensor,
    std::optional<at::Tensor>,
    std::optional<at::Tensor>,
    std::optional<at::Tensor>,
    std::optional<at::Tensor>>
_block_bucketize_sparse_features_xpu(
        const at::Tensor& lengths,
        const at::Tensor& indices,
        const bool bucketize_pos,
        const bool sequence,
        const at::Tensor& block_sizes,
        const std::optional<at::Tensor>& total_num_blocks,
        const int64_t my_size,
        const std::optional<at::Tensor>& weights,
        const std::optional<at::Tensor>& batch_size_per_feature,
        const int64_t max_B,
        const std::optional<std::vector<at::Tensor>>& block_bucketize_pos,
        const bool return_bucket_mapping,
        const bool keep_orig_idx,
        const std::optional<at::Tensor>& keep_orig_idx_per_feature) {

    TENSORS_ON_SAME_SYCL_XPU_IF_NOT_OPTIONAL(lengths, indices);

    SYCL_DEVICE_GUARD(lengths);

    if (total_num_blocks.has_value() &&
            (!block_bucketize_pos.has_value() || block_bucketize_pos.value().empty())) {
        // divisibility check runs on CPU scalars
        TORCH_CHECK(my_size > 0);
        at::Tensor tnb = total_num_blocks.value().cpu();
        AT_DISPATCH_INDEX_TYPES(tnb.scalar_type(), "tnb_check", [&] {
            const index_t* p = tnb.const_data_ptr<index_t>();
            for (int64_t t = 0; t < tnb.numel(); ++t) {
                TORCH_CHECK(
                    p[t] % static_cast<index_t>(my_size) == 0,
                    "block_bucketize_sparse_features: total_num_blocks[", t,
                    "] = ", p[t], " must be a multiple of my_size (", my_size, ")");
            }
        });
    }

    const int64_t lengths_size = lengths.numel();
    const int64_t T = block_sizes.numel();
    const int64_t B = lengths_size / T;
    const int64_t new_lengths_size = lengths_size * my_size;

    auto offsets = fbgemm_xpu::local_complete_cumsum_xpu(lengths.contiguous());
    // offsets is size lengths_size+1; we slice to lengths_size for inclusive sums
    // The kernel uses offsets_data[b_t-1]..offsets_data[b_t] to get row bounds.
    // asynchronous_complete_cumsum_xpu returns [0, c0, c0+c1, ...] (N+1 elements)
    // → offsets_data[b_t] = inclusive cumsum at b_t (i.e. end of row b_t)
    //   offsets_data[b_t-1] = start of row b_t
    // The kernel handles b_t==0 specially.

    auto new_lengths = at::zeros({new_lengths_size}, lengths.options());
    auto new_offsets = at::empty({new_lengths_size}, lengths.options());
    auto new_indices = at::empty_like(indices);
    auto lengths_contig = lengths.contiguous();
    auto indices_contig = indices.contiguous();

    std::optional<at::Tensor> new_weights;
    std::optional<at::Tensor> new_pos;
    std::optional<at::Tensor> unbucketize_permute;
    std::optional<at::Tensor> bucket_mapping;

    // -----------------------------------------------------------------------
    // Prepare concat/offsets for block_bucketize_pos (variable bucket sizes)
    // -----------------------------------------------------------------------
    at::Tensor bbp_concat = at::empty({1}, indices_contig.options());
    at::Tensor bbp_offsets = at::empty({1}, indices_contig.options());
    bool has_bbp = block_bucketize_pos.has_value() &&
                   !block_bucketize_pos.value().empty();

    if (has_bbp) {
        const auto& pos_tensors = block_bucketize_pos.value();
        bbp_concat = at::cat(pos_tensors, 0);
        std::vector<int64_t> sizes;
        sizes.reserve(pos_tensors.size() + 1);
        for (const auto& t_pos : pos_tensors) sizes.push_back(t_pos.numel());
        sizes.push_back(0);
        auto sizes_cpu = at::tensor(sizes, at::TensorOptions().dtype(indices_contig.dtype()).device(at::kCPU));
        // Exclusive cumsum on CPU then move to XPU
        at::Tensor ex = at::zeros({(int64_t)sizes.size()}, sizes_cpu.options());
        for (int64_t i = 1; i < (int64_t)sizes.size(); ++i)
            ex[i] = ex[i-1] + sizes_cpu[i-1];
        bbp_offsets = ex.to(indices_contig.device(), /*non_blocking=*/true);
    }

    // indices_to_lb: stores binary search results for variable bucket mode
    at::Tensor indices_to_lb = at::empty_like(indices_contig);

    // -----------------------------------------------------------------------
    // Prepare variable batch: length_to_feature_idx
    // -----------------------------------------------------------------------
    at::Tensor length_to_feature_idx = at::empty({lengths_size}, lengths.options());
    bool has_variable_batch = batch_size_per_feature.has_value();
    if (has_variable_batch) {
        TORCH_CHECK(max_B > 0);
        auto bsf_contig = batch_size_per_feature.value().contiguous();
        // Build batch_size_offsets (exclusive cumsum)
        at::Tensor bso = xpu_excl_cumsum(bsf_contig);

        sycl::queue& q = c10::xpu::getCurrentXPUStream().queue();
        const int64_t total = max_B * T;
        const int64_t gs = grid_size(total) * kThreads;

        AT_DISPATCH_INDEX_TYPES(
            lengths.scalar_type(), "populate_len_to_feat", [&] {
                using offset_t = index_t;
                q.submit([&](sycl::handler& cgh) {
                    cgh.parallel_for<PopulateLengthToFeatureIdxKernel<offset_t>>(
                        sycl::nd_range<1>(sycl::range<1>(gs), sycl::range<1>(kThreads)),
                        PopulateLengthToFeatureIdxKernel<offset_t>(
                            max_B, T,
                            bsf_contig.data_ptr<offset_t>(),
                            bso.data_ptr<offset_t>(),
                            length_to_feature_idx.data_ptr<offset_t>()));
                });
            });
    }

    sycl::queue& queue = c10::xpu::getCurrentXPUStream().queue();
    const int64_t gs = grid_size(lengths_size) * kThreads;

    if (use_chunked_block_bucketize(indices.numel(), lengths_size, my_size)) {
        const int64_t lengths_sum = indices.numel();
        std::optional<at::Tensor> weights_contig;
        if (weights.has_value()) {
            weights_contig = weights.value().contiguous();
            new_weights = at::empty_like(*weights_contig);
        }
        if (sequence) {
            unbucketize_permute = at::empty({lengths_sum}, indices.options());
        }
        if (return_bucket_mapping) {
            bucket_mapping = at::empty({lengths_sum}, indices.options());
        }
        if (bucketize_pos) {
            new_pos = at::empty_like(indices);
        }
        block_bucketize_chunked_xpu(
            queue, lengths_contig, indices_contig, offsets, block_sizes,
            total_num_blocks, B, my_size, sequence, weights_contig,
            keep_orig_idx_per_feature, keep_orig_idx,
            has_variable_batch ? std::optional<at::Tensor>(length_to_feature_idx) : std::nullopt,
            has_bbp ? std::optional<at::Tensor>(bbp_concat) : std::nullopt,
            has_bbp ? std::optional<at::Tensor>(bbp_offsets) : std::nullopt,
            indices_to_lb, new_lengths, new_indices,
            new_weights, new_pos, unbucketize_permute, bucket_mapping);
        return {new_lengths, new_indices, new_weights, new_pos, unbucketize_permute, bucket_mapping};
    }

    // -----------------------------------------------------------------------
    // Kernel 1: count new_lengths
    // -----------------------------------------------------------------------
    AT_DISPATCH_INDEX_TYPES(
        lengths.scalar_type(), "block_bucketize_lengths_xpu_1", [&] {
            using offset_t = index_t;
            AT_DISPATCH_INDEX_TYPES(
                indices.scalar_type(), "block_bucketize_lengths_xpu_2", [&] {
                    queue.submit([&](sycl::handler& cgh) {
                        cgh.parallel_for<BlockBucketizeCountKernel<offset_t, index_t>>(
                            sycl::nd_range<1>(sycl::range<1>(gs), sycl::range<1>(kThreads)),
                            BlockBucketizeCountKernel<offset_t, index_t>(
                                lengths_size, B, my_size,
                                offsets.data_ptr<offset_t>() + 1, // +1: skip leading 0
                                indices_contig.data_ptr<index_t>(),
                                block_sizes.data_ptr<index_t>(),
                                has_variable_batch
                                    ? length_to_feature_idx.data_ptr<offset_t>()
                                    : static_cast<offset_t*>(nullptr),
                                has_bbp ? bbp_concat.data_ptr<index_t>() : nullptr,
                                has_bbp ? bbp_offsets.data_ptr<index_t>() : nullptr,
                                total_num_blocks.has_value()
                                    ? total_num_blocks.value().data_ptr<index_t>()
                                    : nullptr,
                                new_lengths.data_ptr<offset_t>(),
                                has_bbp ? indices_to_lb.data_ptr<index_t>() : nullptr));
                    });
                });
        });

    // -----------------------------------------------------------------------
    // Build new_offsets = exclusive cumsum of new_lengths
    // -----------------------------------------------------------------------
    new_offsets = xpu_excl_cumsum(new_lengths);

    // -----------------------------------------------------------------------
    // Kernel 2/3: scatter
    // -----------------------------------------------------------------------
    const int64_t lengths_sum = indices.numel();

#define LAUNCH_SCATTER_SEQ_W(bp, rbm)                                           \
    AT_DISPATCH_INDEX_TYPES(                                                    \
        lengths.scalar_type(), "block_bucketize_indices_w_xpu_1", [&] {         \
            using offset_t = index_t;                                           \
            AT_DISPATCH_INDEX_TYPES(                                            \
                indices.scalar_type(), "block_bucketize_indices_w_xpu_2", [&] { \
                    FBGEMM_DISPATCH_FLOAT_AND_DOUBLE(                            \
                        weights_value.scalar_type(),                            \
                        "block_bucketize_indices_w_xpu_3", [&] {                \
                            queue.submit([&](sycl::handler& cgh) {              \
                                cgh.parallel_for<                               \
                                    BlockBucketizeScatterSeqKernel<             \
                                        true, bp, rbm,                          \
                                        offset_t, index_t, scalar_t>>(         \
                                    sycl::nd_range<1>(                          \
                                        sycl::range<1>(gs),                    \
                                        sycl::range<1>(kThreads)),              \
                                    BlockBucketizeScatterSeqKernel<             \
                                        true, bp, rbm,                          \
                                        offset_t, index_t, scalar_t>(          \
                                        lengths_size, B, my_size,               \
                                        offsets.data_ptr<offset_t>() + 1,      \
                                        indices_contig.data_ptr<index_t>(),     \
                                        weights_value_contig.data_ptr<scalar_t>(), \
                                        block_sizes.data_ptr<index_t>(),        \
                                        has_variable_batch                      \
                                            ? length_to_feature_idx.data_ptr<offset_t>() \
                                            : nullptr,                          \
                                        has_bbp ? bbp_concat.data_ptr<index_t>() : nullptr, \
                                        has_bbp ? bbp_offsets.data_ptr<index_t>() : nullptr, \
                                        has_bbp ? indices_to_lb.data_ptr<index_t>() : nullptr, \
                                        total_num_blocks.has_value()            \
                                            ? total_num_blocks.value().data_ptr<index_t>() \
                                            : nullptr,                          \
                                        new_offsets.data_ptr<offset_t>(),       \
                                        new_indices.data_ptr<index_t>(),        \
                                        new_weights.value().data_ptr<scalar_t>(), \
                                        (bp) ? new_pos.value().data_ptr<index_t>() : nullptr, \
                                        unbucketize_permute.value().data_ptr<index_t>(), \
                                        (rbm) ? bucket_mapping.value().data_ptr<index_t>() : nullptr, \
                                        keep_orig_idx_per_feature.has_value()   \
                                            ? keep_orig_idx_per_feature.value().const_data_ptr<bool>() \
                                            : nullptr,                          \
                                        keep_orig_idx));                        \
                            });                                                 \
                        });                                                     \
                });                                                             \
        });

#define LAUNCH_SCATTER_SEQ_NW(bp, rbm)                                          \
    AT_DISPATCH_INDEX_TYPES(                                                    \
        lengths.scalar_type(), "block_bucketize_seq_nw_xpu_1", [&] {            \
            using offset_t = index_t;                                           \
            AT_DISPATCH_INDEX_TYPES(                                            \
                indices.scalar_type(), "block_bucketize_seq_nw_xpu_2", [&] {    \
                    queue.submit([&](sycl::handler& cgh) {                      \
                        cgh.parallel_for<                                       \
                            BlockBucketizeScatterSeqKernel<                     \
                                false, bp, rbm,                                 \
                                offset_t, index_t, NoWeightT>>(           \
                            sycl::nd_range<1>(                                  \
                                sycl::range<1>(gs),                            \
                                sycl::range<1>(kThreads)),                     \
                            BlockBucketizeScatterSeqKernel<                     \
                                false, bp, rbm,                                 \
                                offset_t, index_t, NoWeightT>(            \
                                lengths_size, B, my_size,                      \
                                offsets.data_ptr<offset_t>() + 1,              \
                                indices_contig.data_ptr<index_t>(),             \
                                nullptr,                                        \
                                block_sizes.data_ptr<index_t>(),                \
                                has_variable_batch                              \
                                    ? length_to_feature_idx.data_ptr<offset_t>() \
                                    : nullptr,                                  \
                                has_bbp ? bbp_concat.data_ptr<index_t>() : nullptr, \
                                has_bbp ? bbp_offsets.data_ptr<index_t>() : nullptr, \
                                has_bbp ? indices_to_lb.data_ptr<index_t>() : nullptr, \
                                total_num_blocks.has_value()                    \
                                    ? total_num_blocks.value().data_ptr<index_t>() \
                                    : nullptr,                                  \
                                new_offsets.data_ptr<offset_t>(),               \
                                new_indices.data_ptr<index_t>(),                \
                                nullptr,                                        \
                                (bp) ? new_pos.value().data_ptr<index_t>() : nullptr, \
                                unbucketize_permute.value().data_ptr<index_t>(), \
                                (rbm) ? bucket_mapping.value().data_ptr<index_t>() : nullptr, \
                                keep_orig_idx_per_feature.has_value()           \
                                    ? keep_orig_idx_per_feature.value().const_data_ptr<bool>() \
                                    : nullptr,                                  \
                                keep_orig_idx));                                \
                    });                                                         \
                });                                                             \
        });

#define LAUNCH_SCATTER_POOL_W(bp)                                               \
    AT_DISPATCH_INDEX_TYPES(                                                    \
        lengths.scalar_type(), "block_bucketize_pool_w_xpu_1", [&] {            \
            using offset_t = index_t;                                           \
            AT_DISPATCH_INDEX_TYPES(                                            \
                indices.scalar_type(), "block_bucketize_pool_w_xpu_2", [&] {    \
                    FBGEMM_DISPATCH_FLOAT_AND_DOUBLE(                            \
                        weights_value.scalar_type(),                            \
                        "block_bucketize_pool_w_xpu_3", [&] {                   \
                            queue.submit([&](sycl::handler& cgh) {              \
                                cgh.parallel_for<                               \
                                    BlockBucketizeScatterPooledKernel<          \
                                        true, bp,                               \
                                        offset_t, index_t, scalar_t>>(         \
                                    sycl::nd_range<1>(                          \
                                        sycl::range<1>(gs),                    \
                                        sycl::range<1>(kThreads)),              \
                                    BlockBucketizeScatterPooledKernel<          \
                                        true, bp,                               \
                                        offset_t, index_t, scalar_t>(          \
                                        lengths_size, B, my_size,               \
                                        offsets.data_ptr<offset_t>() + 1,      \
                                        indices_contig.data_ptr<index_t>(),     \
                                        weights_value_contig.data_ptr<scalar_t>(), \
                                        block_sizes.data_ptr<index_t>(),        \
                                        has_variable_batch                      \
                                            ? length_to_feature_idx.data_ptr<offset_t>() \
                                            : nullptr,                          \
                                        has_bbp ? bbp_concat.data_ptr<index_t>() : nullptr, \
                                        has_bbp ? bbp_offsets.data_ptr<index_t>() : nullptr, \
                                        has_bbp ? indices_to_lb.data_ptr<index_t>() : nullptr, \
                                        total_num_blocks.has_value()            \
                                            ? total_num_blocks.value().data_ptr<index_t>() \
                                            : nullptr,                          \
                                        new_offsets.data_ptr<offset_t>(),       \
                                        new_indices.data_ptr<index_t>(),        \
                                        new_weights.value().data_ptr<scalar_t>(), \
                                        (bp) ? new_pos.value().data_ptr<index_t>() : nullptr, \
                                        keep_orig_idx_per_feature.has_value()   \
                                            ? keep_orig_idx_per_feature.value().const_data_ptr<bool>() \
                                            : nullptr,                          \
                                        keep_orig_idx));                        \
                            });                                                 \
                        });                                                     \
                });                                                             \
        });

#define LAUNCH_SCATTER_POOL_NW(bp)                                              \
    AT_DISPATCH_INDEX_TYPES(                                                    \
        lengths.scalar_type(), "block_bucketize_pool_nw_xpu_1", [&] {           \
            using offset_t = index_t;                                           \
            AT_DISPATCH_INDEX_TYPES(                                            \
                indices.scalar_type(), "block_bucketize_pool_nw_xpu_2", [&] {   \
                    queue.submit([&](sycl::handler& cgh) {                      \
                        cgh.parallel_for<                                       \
                            BlockBucketizeScatterPooledKernel<                  \
                                false, bp,                                      \
                                offset_t, index_t, NoWeightT>>(           \
                            sycl::nd_range<1>(                                  \
                                sycl::range<1>(gs),                            \
                                sycl::range<1>(kThreads)),                     \
                            BlockBucketizeScatterPooledKernel<                  \
                                false, bp,                                      \
                                offset_t, index_t, NoWeightT>(            \
                                lengths_size, B, my_size,                       \
                                offsets.data_ptr<offset_t>() + 1,              \
                                indices_contig.data_ptr<index_t>(),             \
                                nullptr,                                        \
                                block_sizes.data_ptr<index_t>(),                \
                                has_variable_batch                              \
                                    ? length_to_feature_idx.data_ptr<offset_t>() \
                                    : nullptr,                                  \
                                has_bbp ? bbp_concat.data_ptr<index_t>() : nullptr, \
                                has_bbp ? bbp_offsets.data_ptr<index_t>() : nullptr, \
                                has_bbp ? indices_to_lb.data_ptr<index_t>() : nullptr, \
                                total_num_blocks.has_value()                    \
                                    ? total_num_blocks.value().data_ptr<index_t>() \
                                    : nullptr,                                  \
                                new_offsets.data_ptr<offset_t>(),               \
                                new_indices.data_ptr<index_t>(),                \
                                nullptr,                                        \
                                (bp) ? new_pos.value().data_ptr<index_t>() : nullptr, \
                                keep_orig_idx_per_feature.has_value()           \
                                    ? keep_orig_idx_per_feature.value().const_data_ptr<bool>() \
                                    : nullptr,                                  \
                                keep_orig_idx));                                \
                    });                                                         \
                });                                                             \
        });

    if (weights.has_value()) {
        at::Tensor weights_value = weights.value();
        at::Tensor weights_value_contig = weights_value.contiguous();
        new_weights = at::empty_like(weights_value);
        if (sequence) {
            unbucketize_permute = at::empty({lengths_sum}, indices.options());
            if (return_bucket_mapping) {
                bucket_mapping = at::empty({lengths_sum}, indices.options());
                if (bucketize_pos) {
                    new_pos = at::empty_like(indices);
                    LAUNCH_SCATTER_SEQ_W(true, true)
                } else {
                    LAUNCH_SCATTER_SEQ_W(false, true)
                }
            } else {
                if (bucketize_pos) {
                    new_pos = at::empty_like(indices);
                    LAUNCH_SCATTER_SEQ_W(true, false)
                } else {
                    LAUNCH_SCATTER_SEQ_W(false, false)
                }
            }
        } else {
            if (return_bucket_mapping) {
                bucket_mapping = at::empty({lengths_sum}, indices.options());
            }
            if (bucketize_pos) {
                new_pos = at::empty_like(indices);
                LAUNCH_SCATTER_POOL_W(true)
            } else {
                LAUNCH_SCATTER_POOL_W(false)
            }
        }
    } else {
        if (sequence) {
            unbucketize_permute = at::empty({lengths_sum}, indices.options());
            if (return_bucket_mapping) {
                bucket_mapping = at::empty({lengths_sum}, indices.options());
                if (bucketize_pos) {
                    new_pos = at::empty_like(indices);
                    LAUNCH_SCATTER_SEQ_NW(true, true)
                } else {
                    LAUNCH_SCATTER_SEQ_NW(false, true)
                }
            } else {
                if (bucketize_pos) {
                    new_pos = at::empty_like(indices);
                    LAUNCH_SCATTER_SEQ_NW(true, false)
                } else {
                    LAUNCH_SCATTER_SEQ_NW(false, false)
                }
            }
        } else {
            if (return_bucket_mapping) {
                bucket_mapping = at::empty({lengths_sum}, indices.options());
            }
            if (bucketize_pos) {
                new_pos = at::empty_like(indices);
                LAUNCH_SCATTER_POOL_NW(true)
            } else {
                LAUNCH_SCATTER_POOL_NW(false)
            }
        }
    }

#undef LAUNCH_SCATTER_SEQ_W
#undef LAUNCH_SCATTER_SEQ_NW
#undef LAUNCH_SCATTER_POOL_W
#undef LAUNCH_SCATTER_POOL_NW

    return {new_lengths, new_indices, new_weights, new_pos, unbucketize_permute, bucket_mapping};
}

// ============================================================================
// Public API – XPU dispatch
// Parameter order matches ops_registry.cpp schema:
//   lengths, indices, bucketize_pos, sequence, block_sizes, my_size,
//   weights, batch_size_per_feature, max_B, block_bucketize_pos,
//   keep_orig_idx, total_num_blocks, keep_orig_idx_per_feature
// ============================================================================

static std::tuple<
    at::Tensor,
    at::Tensor,
    std::optional<at::Tensor>,
    std::optional<at::Tensor>,
    std::optional<at::Tensor>>
block_bucketize_sparse_features_xpu(
        const at::Tensor& lengths,
        const at::Tensor& indices,
        const bool bucketize_pos,
        const bool sequence,
        const at::Tensor& block_sizes,
        const int64_t my_size,
        const std::optional<at::Tensor>& weights,
        const std::optional<at::Tensor>& batch_size_per_feature,
        const int64_t max_B,
        const std::optional<std::vector<at::Tensor>>& block_bucketize_pos,
        const bool keep_orig_idx,
        const std::optional<at::Tensor>& total_num_blocks,
        const std::optional<at::Tensor>& keep_orig_idx_per_feature) {
    auto [nl, ni, nw, np, up, _] = _block_bucketize_sparse_features_xpu(
        lengths, indices, bucketize_pos, sequence, block_sizes,
        total_num_blocks, my_size, weights, batch_size_per_feature,
        max_B, block_bucketize_pos, false, keep_orig_idx, keep_orig_idx_per_feature);
    return {nl, ni, nw, np, up};
}

static std::tuple<
    at::Tensor,
    at::Tensor,
    std::optional<at::Tensor>,
    std::optional<at::Tensor>,
    std::optional<at::Tensor>,
    std::optional<at::Tensor>>
block_bucketize_sparse_features_inference_xpu(
        const at::Tensor& lengths,
        const at::Tensor& indices,
        const bool bucketize_pos,
        const bool sequence,
        const at::Tensor& block_sizes,
        const int64_t my_size,
        const std::optional<at::Tensor>& weights,
        const std::optional<at::Tensor>& batch_size_per_feature,
        const int64_t max_B,
        const std::optional<std::vector<at::Tensor>>& block_bucketize_pos,
        const bool return_bucket_mapping,
        const bool keep_orig_idx,
        const std::optional<at::Tensor>& total_num_blocks,
        const std::optional<at::Tensor>& keep_orig_idx_per_feature) {
    return _block_bucketize_sparse_features_xpu(
        lengths, indices, bucketize_pos, sequence, block_sizes,
        total_num_blocks, my_size, weights, batch_size_per_feature,
        max_B, block_bucketize_pos, return_bucket_mapping,
        keep_orig_idx, keep_orig_idx_per_feature);
}

// ============================================================================
// populate_bucketized_permute – XPU
// ============================================================================

static at::Tensor populate_bucketized_permute_xpu(
        const at::Tensor& lengths,
        const at::Tensor& bucketized_lengths,
        const at::Tensor& bucket_mapping) {
    TENSORS_ON_SAME_SYCL_XPU_IF_NOT_OPTIONAL(
        lengths, bucketized_lengths, bucket_mapping);
    SYCL_DEVICE_GUARD(lengths);
    const auto lengths_contig = lengths.expect_contiguous();
    const auto bucketized_lengths_contig = bucketized_lengths.expect_contiguous();
    const auto bucket_mapping_contig = bucket_mapping.expect_contiguous();

    at::Tensor bucketized_permute = at::empty_like(*bucket_mapping_contig);
    at::Tensor offsets = fbgemm_xpu::local_complete_cumsum_xpu(*lengths_contig);
    at::Tensor bucketized_offsets = fbgemm_xpu::local_complete_cumsum_xpu(*bucketized_lengths_contig);
    // Both are N+1 tensors; slice to get exclusive prefix [0..N-1]
    at::Tensor offsets_excl = offsets.slice(0, 0, lengths.numel());
    at::Tensor bucketized_offsets_work = bucketized_offsets.slice(0, 0, bucketized_lengths.numel()).clone();

    const int64_t lengths_size = lengths.numel();
    sycl::queue& queue = c10::xpu::getCurrentXPUStream().queue();
    const int64_t gs = grid_size(lengths_size) * kThreads;

    AT_DISPATCH_INDEX_TYPES(
        lengths.scalar_type(), "populate_bucketized_permute_xpu_1", [&] {
            using offset_t = index_t;
            AT_DISPATCH_INDEX_TYPES(
                bucket_mapping_contig->scalar_type(),
                "populate_bucketized_permute_xpu_2",
                [&] {
                    queue.submit([&](sycl::handler& cgh) {
                        cgh.parallel_for<PopulateBucketizedPermuteKernel<offset_t, index_t>>(
                            sycl::nd_range<1>(sycl::range<1>(gs), sycl::range<1>(kThreads)),
                            PopulateBucketizedPermuteKernel<offset_t, index_t>(
                                lengths_contig->data_ptr<offset_t>(),
                                offsets_excl.data_ptr<offset_t>(),
                                bucketized_offsets_work.data_ptr<offset_t>(),
                                bucket_mapping_contig->data_ptr<index_t>(),
                                bucketized_permute.data_ptr<index_t>(),
                                lengths_size));
                    });
                });
        });

    return bucketized_permute;
}

// ============================================================================
// Dispatch registration
// ============================================================================

TORCH_LIBRARY_IMPL(fbgemm, XPU, m) {
    m.impl("block_bucketize_sparse_features",
           &block_bucketize_sparse_features_xpu);
    m.impl("block_bucketize_sparse_features_inference",
           &block_bucketize_sparse_features_inference_xpu);
    m.impl("populate_bucketized_permute",
           &populate_bucketized_permute_xpu);
}

} // namespace fbgemm_xpu
