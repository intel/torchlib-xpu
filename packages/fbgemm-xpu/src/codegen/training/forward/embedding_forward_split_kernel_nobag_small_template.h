/*
 * Copyright (c) Meta Platforms, Inc. and affiliates. All rights reserved.
 * Copyright (c) 2026 Intel Corporation. All Rights Reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */

{#
// @lint-ignore LINTIGNORE
// @lint-ignore-every CLANGFORMAT
// clang-format off
#}

{%- set mdesc = "dense" if dense else "split" %}

////////////////////////////////////////////////////////////////////////////////
// SYCL PORT MAPPING TO FBGEMM CUDA SOURCE - FORWARD KERNELS
////////////////////////////////////////////////////////////////////////////////
//
// This file contains SYCL port of FBGEMM {{ mdesc }} embedding lookup forward 
// unweighted small kernel.
//
// ORIGINAL CUDA SOURCE:
//   Template: fbgemm_gpu/codegen/training/forward/embedding_forward_split_kernel_nobag_small_template.cu
//   Generated Source: fbgemm_gpu/_skbuild/linux-x86_64-3.10/cmake-build/gen_embedding_forward_{{ mdesc }}_unweighted_nobag_kernel_small.cu
//
// KERNEL MAPPING:
//   {{ mdesc | capitalize }}EmbeddingNobagCodegenForwardUnweightedSmallKernel
//     → {{ mdesc }}_embedding_nobag_codegen_forward_unweighted_small_kernel (CUDA)
//
// DESCRIPTION:
//   Optimized forward kernel for small embedding dimensions (D <= 32).
//   Uses sub-group shuffle operations for efficient small-dimension lookups.
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
    typename index_t,
    size_t kThreadGroupSize>
    class {{ mdesc | capitalize }}EmbeddingNobagCodegenForwardUnweightedSmallKernel {
        public:
            {{ mdesc | capitalize }}EmbeddingNobagCodegenForwardUnweightedSmallKernel(
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
            {%- endif %}
            mutable at::PackedTensorAccessor64<output_t, 2, RestrictPtrTraits> output_;
    };

    template <
    typename emb_t,
    typename cache_t,
    typename output_t,
    typename index_t,
    size_t kThreadGroupSize>
    inline void {{ mdesc | capitalize }}EmbeddingNobagCodegenForwardUnweightedSmallKernel<emb_t, cache_t, output_t, index_t, kThreadGroupSize>
    ::operator()(const sycl::nd_item<2>& item) const {
        auto b_t = item.get_group(0) * item.get_local_range(0) +
                item.get_local_id(0);
        if (static_cast<int64_t>(b_t) >= offsets_.size(0) - 1) {
            return;
        }

        int32_t t;
        int32_t b;
        fd_B_.DivMod(b_t, &t, &b);

        const auto indices_start = offsets_[b_t];
        const auto L = offsets_[b_t + 1] - indices_start;

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

        int32_t D_emb = D_;

        const int32_t lane_id = static_cast<int32_t>(item.get_local_id(1));
        const int32_t group_start =
            (lane_id / static_cast<int32_t>(kThreadGroupSize)) * static_cast<int32_t>(kThreadGroupSize);
        const int32_t group_end = group_start + static_cast<int32_t>(kThreadGroupSize);
        const int32_t d = (lane_id % static_cast<int32_t>(kThreadGroupSize)) * 4;
        const uint32_t kSubGroupSize = item.get_sub_group().get_local_range().get(0);

        for (int32_t l_start = 0; l_start < L; l_start += kSubGroupSize) {
            const int32_t l = l_start + lane_id;
            int64_t idx = l < L ? indices_[indices_start + l] : 0;
            {%- if not dense %}
            const int32_t cache_idx =
                (placement == PlacementType::MANAGED_CACHING && l < L)
                    ? lxu_cache_locations_[indices_start + l] : 0;
            {%- endif %}

            for (int32_t j = group_start; j < group_end && l_start + j < L; ++j) {
                int64_t idx_j = sycl::select_from_group(item.get_sub_group(), idx, j);
                int64_t output_j = indices_start + l_start + j;
                {%- if not dense %}
                const int32_t cache_idx_j = sycl::select_from_group(item.get_sub_group(), cache_idx, j);
                {%- endif %}

                auto weight_row_emb = WeightRowAccessor<emb_t, cache_t>(
                    &weights[idx_j * D_emb],
                    D_
                );

                if (d < D_) {
                    {%- if not dense %}
                    if (placement == PlacementType::MANAGED_CACHING &&
                        cache_idx_j != kCacheLocationMissing) {
                        const cache_t* cache_weights;
                        cache_weights = reinterpret_cast<const cache_t*>(
                            &lxu_cache_weights_[cache_idx_j][0]);

                        auto weight_row_cache = WeightRowAccessor<cache_t, cache_t>(cache_weights, D_);
                        Vec4T<cache_t> weight = weight_row_cache.load(d);
                        weight.store(&output_[output_j][d]);
                    } else {
                        Vec4T<cache_t> weight = weight_row_emb.load(d);
                        weight.store(&output_[output_j][d]);
                    }
                    {%- else %}
                    Vec4T<cache_t> weight = weight_row_emb.load(d);
                    // output is 2D
                    weight.store(&output_[output_j][d]);
                    {%- endif %}
                }
            }
        }
    }

} // namespace fbgemm_xpu
