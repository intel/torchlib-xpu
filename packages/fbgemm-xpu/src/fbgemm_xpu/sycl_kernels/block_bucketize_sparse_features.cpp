/*
 * Copyright (c) Meta Platforms, Inc. and affiliates. All rights reserved.
 * Copyright (c) 2026 Intel Corporation. All Rights Reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */

/*
 * SYCL/XPU Implementation of block_bucketize_sparse_features
 *
 * SYCL/XPU implementation of block_bucketize_sparse_features,
 * block_bucketize_sparse_features_inference, and populate_bucketized_permute.
 *
 * Design: one work-item per (b_t) row for both the count and scatter
 * phases. This avoids per-element atomics on new_lengths/new_offsets
 * because each work-item owns a disjoint column of those arrays.
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

#include "block_bucketize_sparse_features.h"

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

// Kernel 1 – count new_lengths
// One work-item per b_t; serial inner loop over its indices.
// No atomics needed: each work-item is the sole writer to column b_t.
// ============================================================================

template <typename offset_t, typename index_t>
void BlockBucketizeCountKernel<offset_t, index_t>::operator()(
        const sycl::nd_item<1>& item) const {
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
void BlockBucketizeScatterSeqKernel<
        has_weight, bucketize_pos_flag, return_bucket_mapping,
        offset_t, index_t, scalar_t>::operator()(
        const sycl::nd_item<1>& item) const {
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
void BlockBucketizeScatterPooledKernel<
        has_weight, bucketize_pos_flag,
        offset_t, index_t, scalar_t>::operator()(
        const sycl::nd_item<1>& item) const {
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


// ============================================================================
// Kernel 4 – populate_bucketized_permute
// One work-item per b_t; serial inner loop.
// ============================================================================

template <typename offset_t, typename index_t>
void PopulateBucketizedPermuteKernel<offset_t, index_t>::operator()(
        const sycl::nd_item<1>& item) const {
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


// ============================================================================
// Helpers: variable-batch length_to_feature_idx population
// ============================================================================

template <typename offset_t>
void PopulateLengthToFeatureIdxKernel<offset_t>::operator()(
        const sycl::nd_item<1>& item) const {
    const int64_t b_t = item.get_global_id(0);
    const int64_t t = b_t / max_B_;
    const int64_t b = b_t % max_B_;
    if (t >= T_ || static_cast<offset_t>(b) >= batch_size_per_feature_[t]) return;
    length_to_feature_idx_[batch_size_offsets_[t] + b] = static_cast<offset_t>(t);
}


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

    TORCH_INTERNAL_ASSERT(
        lengths.device().type() == at::DeviceType::XPU,
        "block_bucketize_sparse_features_xpu: lengths must be on XPU");

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
