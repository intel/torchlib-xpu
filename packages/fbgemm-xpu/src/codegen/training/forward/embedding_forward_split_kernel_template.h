/*
 * Copyright (c) Meta Platforms, Inc. and affiliates. All rights reserved.
 * Copyright (c) 2026 Intel Corporation. All Rights Reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */

{#
// @lint-ignore LINTIGNORE
// @lint-ignore-every CLANGFORMAT
// clang-format off
// Note: clang-format off doesn't work with this templaterized code,
// so we need to keep lint-ignore-every.
#}

{%- set mdesc = "dense" if dense else "split" %}

////////////////////////////////////////////////////////////////////////////////
// SYCL PORT MAPPING TO FBGEMM CUDA SOURCE - FORWARD KERNELS
////////////////////////////////////////////////////////////////////////////////
//
// This file contains SYCL port of FBGEMM {{ mdesc }} embedding lookup forward 
// unweighted kernel.
//
// ORIGINAL CUDA SOURCE:
//   Template: fbgemm_gpu/codegen/training/forward/embedding_forward_split_kernel_template.cu
//   Generated Source: fbgemm_gpu/_skbuild/linux-x86_64-3.10/cmake-build/gen_embedding_forward_{{ mdesc }}_unweighted_nobag_kernel.cu
//
// KERNEL MAPPING:
//   {{ mdesc | capitalize }}EmbeddingNobagCodegenForwardUnweightedKernel
//     → {{ mdesc }}_embedding_nobag_codegen_forward_unweighted_kernel (CUDA)
//
// DESCRIPTION:
//   Main forward kernel for no-bag embeddings (sequence embeddings).
{%- if not dense %}
//   Supports cache-aware lookups with LXU cache for UVM-managed tables.
{%- else %}
//   Each thread retrieves one embedding vector and copies to output.
{%- endif %}
//
////////////////////////////////////////////////////////////////////////////////


#pragma once

{%- if dense %}
#include <cassert>
#include <cstdlib>
{%- endif %}

#include <sycl/sycl.hpp>
#include <c10/xpu/XPUStream.h>

{%- if dense %}
#include <ATen/ATen.h>
{%- endif %}
#include <ATen/Operators.h>
{%- if dense %}
#include <ATen/core/TensorAccessor.h>
{%- endif %}
#include <torch/all.h>
#include <torch/library.h>
#include <torch/csrc/autograd/record_function_ops.h>
#include <ATen/native/StridedRandomAccessor.h>

{%- if dense %}
#include "../fbgemm_utils/tensor_utils.h"
{%- endif %}
#include "../fbgemm_utils/utils.h"
#include "../fbgemm_utils/weight_row.h"
{%- if not dense %}
#include "../fbgemm_utils/tensor_utils.h"
#include "../fbgemm_utils/split_embeddings_cache_xpu.h"
{%- endif %}

using Tensor = at::Tensor;

using at::native::RestrictPtrTraits;

namespace fbgemm_xpu {

    template <
    typename emb_t,
    typename cache_t,
    typename output_t,
    {%- if not dense %}
    bool use_lxu_cache,
    {%- endif %}
    typename index_t,
    size_t kThreadGroupSize>
    class {{ mdesc | capitalize }}EmbeddingNobagCodegenForwardUnweightedKernel {
        public:
            {{ mdesc | capitalize }}EmbeddingNobagCodegenForwardUnweightedKernel(
                const at::PackedTensorAccessor64<emb_t, 1, RestrictPtrTraits> dev_weights,
                {%- if not dense %}
                const at::PackedTensorAccessor64<emb_t, 1, RestrictPtrTraits> uvm_weights,
                const at::PackedTensorAccessor64<cache_t, 2, RestrictPtrTraits> lxu_cache_weights,
                const at::PackedTensorAccessor32<int32_t, 1, RestrictPtrTraits> weights_placements,
                {%- endif %}
                const at::PackedTensorAccessor32<int64_t, 1, RestrictPtrTraits> weights_offsets,
                int64_t D,
                FixedDivisor fd_B,
                const at::PackedTensorAccessor32<index_t, 1, RestrictPtrTraits> indices,
                const at::PackedTensorAccessor32<index_t, 1, RestrictPtrTraits> offsets,
                {%- if not dense %}
                const at::PackedTensorAccessor32<int32_t, 1, RestrictPtrTraits> lxu_cache_locations,
                const int32_t* lxu_cache_conflict_misses,
                {%- endif %}
                at::PackedTensorAccessor64<output_t, 2, RestrictPtrTraits> output
            ): dev_weights_(dev_weights),
               {%- if not dense %}
               uvm_weights_(uvm_weights),
               lxu_cache_weights_(lxu_cache_weights),
               weights_placements_(weights_placements),
               {%- endif %}
               weights_offsets_(weights_offsets),
               D_(D),
               fd_B_(fd_B),
               indices_(indices),
               offsets_(offsets),
               {%- if not dense %}
               lxu_cache_locations_(lxu_cache_locations),
               lxu_cache_conflict_misses_(lxu_cache_conflict_misses),
               {%- endif %}
               output_(output) {}

            void operator()(const sycl::nd_item<2>& item) const;

        private:
            const at::PackedTensorAccessor64<emb_t, 1, RestrictPtrTraits> dev_weights_;
            {%- if not dense %}
            const at::PackedTensorAccessor64<emb_t, 1, RestrictPtrTraits> uvm_weights_;
            const at::PackedTensorAccessor64<cache_t, 2, RestrictPtrTraits> lxu_cache_weights_;
            const at::PackedTensorAccessor32<int32_t, 1, RestrictPtrTraits> weights_placements_;
            {%- endif %}
            const at::PackedTensorAccessor32<int64_t, 1, RestrictPtrTraits> weights_offsets_;
            const int64_t D_;
            FixedDivisor fd_B_;
            const at::PackedTensorAccessor32<index_t, 1, RestrictPtrTraits> indices_;
            const at::PackedTensorAccessor32<index_t, 1, RestrictPtrTraits> offsets_;
            {%- if not dense %}
            const at::PackedTensorAccessor32<int32_t, 1, RestrictPtrTraits> lxu_cache_locations_;
            const int32_t* lxu_cache_conflict_misses_;
            {%- endif %}
            mutable at::PackedTensorAccessor64<output_t, 2, RestrictPtrTraits> output_;
    };

    template <
    typename emb_t,
    typename cache_t,
    typename output_t,
    {%- if not dense %}
    bool use_lxu_cache,
    {%- endif %}
    typename index_t,
    size_t kThreadGroupSize>
    inline void {{ mdesc | capitalize }}EmbeddingNobagCodegenForwardUnweightedKernel<emb_t, cache_t, output_t, {%- if not dense %}use_lxu_cache, {%- endif %}index_t, kThreadGroupSize>
    ::operator()(const sycl::nd_item<2>& item) const {

        const auto threadIdx_x = item.get_local_id(1);
        const auto threadIdx_y = item.get_local_id(0);
        const auto blockIdx_x = item.get_group(0);
        const auto blockDim_y = item.get_local_range(0);
        const auto gridDim_x = item.get_group_range(0);
        const auto sg = item.get_sub_group();

        // Determine the linearized warp ID, and exit early if needed
        const auto total_B = offsets_.size(0) - 1;
        // Since we place a limit on the grid size, we need to perform grid-striding
        for (auto b_t = blockIdx_x * blockDim_y + threadIdx_y; b_t < total_B; b_t += blockDim_y * gridDim_x) {

            // Determine the Table and Training Example IDs
            int32_t t;  // Table ID
            int32_t b;  // Training Example ID
            fd_B_.DivMod(b_t, &t, &b);

            // Determine the number of indices Vec4(pooling factor) to look up within the bag
            overflow_safe_int_t indices_start = offsets_[b_t];
            int32_t L = offsets_[b_t + 1] - indices_start;

            // Get the offsets of the embedding dimensions of the tables and determine D

            // From the Table ID, fetch its weight tensor offset, locate that position
            // in the input weights tensor, and set the weights table pointer
            const auto weights_offset = weights_offsets_[t];
            const emb_t* __restrict__ weights;
            {%- if not dense %}
            const auto placement = static_cast<PlacementType>(weights_placements_[t]);

            if (placement == PlacementType::DEVICE) {
                weights = &dev_weights_[weights_offset];
            } else {
                weights = &uvm_weights_[weights_offset];
            }
            {%- else %}
            weights = &dev_weights_[weights_offset];
            {%- endif %}

            // D is computed in the bag case or provided as function arg in the nobag case
            // (nobag only supports the case where the embedding dimensions are the same for all tables)
            int32_t D_emb = D_;

            {%- if not dense %}
            if constexpr (!use_lxu_cache) {
                // If use_lxu_cache is false, then the cache conflict miss rate is
                // effectively 100%
                // Iterate over each kThreadGroupSize-sized subset of L indices in the bag
                for (int32_t l_start = 0; l_start < L; l_start += kThreadGroupSize) {
                    // Determine the L index that this thread will load data from in cooperative load
                    auto l = l_start + threadIdx_x;
                    // Cooperatively load the indices
                    const overflow_safe_int_t idx = l < L ? indices_[indices_start + l] : 0;
                    // If idx is loaded
                    const auto offset_idx = idx * D_emb;
                    // Iterate over kThreadGroupSize indices
                    for (auto j = 0; j < kThreadGroupSize && l_start + j < L; ++j) {
                        // Load index from thread j in the group
                        [[maybe_unused]] auto offset_idx_j = sycl::select_from_group(sg, offset_idx, j);
                        overflow_safe_int_t output_j = indices_start + l_start + j;

                        const auto weights_row = WeightRowAccessor
                            <
                                emb_t,
                                cache_t
                            >(
                            &weights[offset_idx_j], // Load from the embedding table
                            D_);
                        for (int32_t i = 0; i < D_; i += kThreadGroupSize * kVecWidth) {
                            const auto d = i + threadIdx_x * kVecWidth;
                            if (d < D_) {
                                // Since there is no pooling, simply copy the weights to output
                                const auto weights_slice = weights_row.load(d);
                                // output is 2D
                                weights_slice.store(&output_[output_j][d]);
                            }
                        }
                        
                    }
                }

            } else {
                if (placement != PlacementType::MANAGED_CACHING) {
                    // Load every row from HBM or UVM
                    // Iterate over each kThreadGroupSize-sized subset of L indices in the bag
                    for (int32_t l_start = 0; l_start < L; l_start += kThreadGroupSize) {
                        // Determine the L index that this thread will load data from in cooperative load
                        auto l = l_start + threadIdx_x;
                        // Cooperatively load the indices
                        const overflow_safe_int_t idx = l < L ? indices_[indices_start + l] : 0;
                        // If idx is loaded
                        const auto offset_idx = idx * D_emb;
                        // Iterate over kThreadGroupSize indices
                        for (auto j = 0; j < kThreadGroupSize && l_start + j < L; ++j) {
                            // Load index from thread j in the group
                            [[maybe_unused]] auto offset_idx_j = sycl::select_from_group(sg, offset_idx, j);
                            overflow_safe_int_t output_j = indices_start + l_start + j;

                            const auto weights_row = WeightRowAccessor
                                <
                                    emb_t,
                                    cache_t
                                >(
                                &weights[offset_idx_j], // Load from the embedding table
                                D_);
                            for (int32_t i = 0; i < D_; i += kThreadGroupSize * kVecWidth) {
                                const auto d = i + threadIdx_x * kVecWidth;
                                if (d < D_) {
                                    // Since there is no pooling, simply copy the weights to output
                                    const auto weights_slice = weights_row.load(d);
                                    // output is 2D
                                    weights_slice.store(&output_[output_j][d]);
                                }
                            }
                        }
                    }
                } else if (lxu_cache_conflict_misses_ && *lxu_cache_conflict_misses_ == 0) {
                    // If the UVM cache stats tensor is valid and tell us there are no
                    // conflict unique misses, then the miss rate is effectively 0%
                        
                    // Iterate over each kThreadGroupSize-sized subset of L indices in the bag
                    for (int32_t l_start = 0; l_start < L; l_start += kThreadGroupSize) {
                        // Determine the L index that this thread will load data from in cooperative load
                        auto l = l_start + threadIdx_x;
                        // Cooperatively load the cache's indices
                        [[maybe_unused]] int32_t cache_idx = (use_lxu_cache && placement == PlacementType::MANAGED_CACHING && l < L) ? lxu_cache_locations_[indices_start + l] : 0;
                        // Iterate over kThreadGroupSize indices
                        for (auto j = 0; j < kThreadGroupSize && l_start + j < L; ++j) {
                            overflow_safe_int_t output_j = indices_start + l_start + j;
                            // Load cache's index from thread j in the group
                            [[maybe_unused]] int32_t cache_idx_j
                                = use_lxu_cache ? sycl::select_from_group(sg, cache_idx, j) : 0;
                                
                            const cache_t* cache_weights = reinterpret_cast<const cache_t*>(
                                &lxu_cache_weights_[cache_idx_j][0]);
                            const auto weights_row = WeightRowAccessor
                                <
                                    cache_t,
                                    cache_t
                                >(
                                cache_weights, // Load from the cache
                                D_);
                            for (int32_t i = 0; i < D_; i += kThreadGroupSize * kVecWidth) {
                                const auto d = i + threadIdx_x * kVecWidth;
                                if (d < D_) {
                                    // Since there is no pooling, simply copy the weights to output
                                    const auto weights_slice = weights_row.load(d);
                                    // output is 2D
                                    weights_slice.store(&output_[output_j][d]);
                                }
                            }
                        }
                    }
                } else {
                    // Else, the cache conflict miss rate is mixed
        
                    
                    // Iterate over each kThreadGroupSize-sized subset of L indices in the bag
                    for (int32_t l_start = 0; l_start < L; l_start += kThreadGroupSize) {
                        // Determine the L index that this thread will load data from in cooperative load
                        auto l = l_start + threadIdx_x;
                        // Cooperatively load the indices
                        const overflow_safe_int_t idx = l < L ? indices_[indices_start + l] : 0;
                        // If idx is loaded
                        const auto offset_idx = idx * D_emb;
                        // Cooperatively load the cache's indices
                        [[maybe_unused]] int32_t cache_idx = (use_lxu_cache && placement == PlacementType::MANAGED_CACHING && l < L) ? lxu_cache_locations_[indices_start + l] : 0;
                        // Iterate over kThreadGroupSize indices
                        for (auto j = 0; j < kThreadGroupSize && l_start + j < L; ++j) {
                            // Load index from thread j in the group
                            [[maybe_unused]] auto offset_idx_j = sycl::select_from_group(sg, offset_idx, j);
                            overflow_safe_int_t output_j = indices_start + l_start + j;
                            // Load cache's index from thread j in the group
                            [[maybe_unused]] int32_t cache_idx_j
                                = use_lxu_cache ? sycl::select_from_group(sg, cache_idx, j) : 0;

                            if (placement == PlacementType::MANAGED_CACHING
                                && cache_idx_j != kCacheLocationMissing
                            ) {
                                const cache_t* cache_weights = reinterpret_cast<const cache_t*>(
                                    &lxu_cache_weights_[cache_idx_j][0]);
                                const auto weights_row = WeightRowAccessor
                                    <
                                        cache_t,
                                        cache_t
                                    >(
                                    cache_weights, // Load from the cache
                                    D_);
                                for (int32_t i = 0; i < D_; i += kThreadGroupSize * kVecWidth) {
                                    const auto d = i + threadIdx_x * kVecWidth;
                                    if (d < D_) {
                                        // Since there is no pooling, simply copy the weights to output
                                        const auto weights_slice = weights_row.load(d);
                                        // output is 2D
                                        weights_slice.store(&output_[output_j][d]);
                                    }
                                }
                            } else {
                                const auto weights_row = WeightRowAccessor
                                    <
                                        emb_t,
                                        cache_t
                                    >(
                                    &weights[offset_idx_j], // Load from the embedding table
                                    D_);
                                for (int32_t i = 0; i < D_; i += kThreadGroupSize * kVecWidth) {
                                    const auto d = i + threadIdx_x * kVecWidth;
                                    if (d < D_) {
                                        // Since there is no pooling, simply copy the weights to output
                                        const auto weights_slice = weights_row.load(d);
                                        // output is 2D
                                        weights_slice.store(&output_[output_j][d]);
                                    }
                                }
                            }
                        }
                    }
                }
            }
            {%- else %}
            // Iterate over each kThreadGroupSize-sized subset of L indices in the bag
            for (int32_t l_start = 0; l_start < L; l_start += kThreadGroupSize) {
                // Determine the L index that this thread will load data from in cooperative load
                auto l = l_start + threadIdx_x;
                // Cooperatively load the indices
                const overflow_safe_int_t idx = l < L ? indices_[indices_start + l] : 0;
                // If idx is loaded
                const auto offset_idx = idx * D_emb;

                // Iterate over kThreadGroupSize indices
                for (auto j = 0; j < kThreadGroupSize && l_start + j < L; ++j) {
                    // Broadcast value from thread j in sub-group to all threads
                    auto offset_idx_j = sycl::select_from_group(sg, offset_idx, j);
                    overflow_safe_int_t output_j = indices_start + l_start + j;

                    const auto weights_row = WeightRowAccessor
                        <
                            emb_t,
                            cache_t
                        >(
                        &weights[offset_idx_j], // Load from the embedding table
                        D_);
                    
                    for (int32_t i = 0; i < D_; i += kThreadGroupSize * kVecWidth) {
                        const auto d = i + threadIdx_x * kVecWidth;

                        if (d < D_) {
                            // Since there is no pooling, simply copy the weights to output
                            const auto weights_slice = weights_row.load(d);
                            // output is 2D
                            weights_slice.store(&output_[output_j][d]);
                        }
                    }
                }
            }
            {%- endif %}
        } // for b_t
    }

} // namespace fbgemm_xpu
