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
// SYCL PORT MAPPING TO FBGEMM CUDA SOURCE - FORWARD HOST DISPATCH
////////////////////////////////////////////////////////////////////////////////
//
// This file contains the SYCL port of the FBGEMM {{ mdesc }} embedding forward
// host dispatch function for no-bag (sequence) unweighted lookups.
//
// ORIGINAL CUDA SOURCE:
//   Template: fbgemm_gpu/codegen/training/forward/embedding_forward_split_template.cu
//   Generated Source: fbgemm_gpu/_skbuild/linux-x86_64-3.10/cmake-build/gen_embedding_forward_{{ mdesc }}_unweighted_codegen_cuda.cu
//
// HOST FUNCTION MAPPING:
//   {{ mdesc }}_embedding_nobag_forward_unweighted_xpu
//     → {{ mdesc }}_embedding_nobag_codegen_forward_unweighted_cuda (CUDA)
//
// KERNEL DISPATCH STRATEGY:
//   Small kernel (D <= 32):
//     {{ mdesc | capitalize }}EmbeddingNobagCodegenForwardUnweightedSmallKernel
//       → {{ mdesc }}_embedding_nobag_codegen_forward_unweighted_small_kernel (CUDA)
//       Source: gen_embedding_forward_{{ mdesc }}_unweighted_nobag_kernel_small.h
//
//   General kernel (D > 32):
//     {{ mdesc | capitalize }}EmbeddingNobagCodegenForwardUnweightedKernel
//       → {{ mdesc }}_embedding_nobag_codegen_forward_unweighted_kernel (CUDA)
//       Source: gen_embedding_forward_{{ mdesc }}_unweighted_nobag_kernel.h
//
// DESCRIPTION:
//   Host function for no-bag (sequence) embedding forward pass without pooling.
//   For each index in the input, copies the corresponding embedding row directly
//   to the output tensor.
//   Selects between the small-D-optimized kernel (D <= 32) and the general kernel
//   (D > 32) at runtime based on the embedding dimension.
{%- if not dense %}
//   Supports cache-aware lookups via the LXU cache for UVM-managed tables.
//   Dispatches the general kernel with use_lxu_cache=true/false based on whether
//   lxu_cache_weights is populated.
{%- endif %}
//
////////////////////////////////////////////////////////////////////////////////

#include <cassert>
#include <cstdlib>

#include <sycl/sycl.hpp>
#include <c10/xpu/XPUStream.h>

#include <ATen/ATen.h>
#include <ATen/Operators.h>
#include <ATen/core/TensorAccessor.h>
#include <ATen/native/StridedRandomAccessor.h>

#include "../fbgemm_utils/dispatch_macros.h"
#include "../fbgemm_utils/tensor_utils.h"
#include "../fbgemm_utils/utils.h"
#include "../fbgemm_utils/weight_row.h"
{%- if not dense %}
#include "../fbgemm_utils/split_embeddings_cache_xpu.h"
{%- endif %}

#include <torch/library.h>

#include "gen_embedding_forward_{{ mdesc }}_unweighted_nobag_kernel_small.h"
#include "gen_embedding_forward_{{ mdesc }}_unweighted_nobag_kernel.h"

using Tensor = at::Tensor;

namespace fbgemm_xpu {

    {%- if not dense %}
    #define DISPATCH_KERNEL_FOR_CACHE_CASE(CACHE_CASE_, ...)                       \
    [&] {                                                                        \
        if (CACHE_CASE_ == false) {                                      \
        constexpr auto use_cache_t = false;                            \
        return __VA_ARGS__();                                                    \
        }                                                                          \
        if (CACHE_CASE_ == true) {                                      \
        constexpr auto use_cache_t = true;                            \
        return __VA_ARGS__();                                                    \
        }                                                                          \
        return;                                                                    \
    }()
    {%- endif %}

    Tensor {{ mdesc }}_embedding_nobag_forward_unweighted_xpu(
        const Tensor& dev_weights,
        {%- if not dense %}
        const Tensor& uvm_weights,
        const Tensor& lxu_cache_weights,
        const Tensor& weights_placements,
        {%- endif %}
        const Tensor& weights_offsets,
        const c10::SymInt D_,
        const Tensor& indices,
        const Tensor& offsets,
        {%- if not dense %}
        const Tensor& lxu_cache_locations,
        const Tensor& uvm_cache_stats,
        {%- endif %}
        const int64_t output_dtype,
        const bool is_experimental
    ) {
        const int64_t D = D_.guard_int(__FILE__, __LINE__);

        TENSORS_ON_SAME_SYCL_XPU_IF_NOT_OPTIONAL(
            {%- if not dense %}
            uvm_weights,
            lxu_cache_weights,
            weights_placements,
            {%- endif %}
            weights_offsets,
            indices,
            offsets,
            {%- if not dense %}
            lxu_cache_locations,
            {%- endif %}
            dev_weights
        );

        SYCL_DEVICE_GUARD(dev_weights);
        int32_t total_L = indices.numel();
        int32_t T = weights_offsets.numel();
        TORCH_CHECK_GT(T, 0);
        {%- if not dense %}
        // offsets = [B x T  + 1]
        {%- endif %}
        const auto total_B = offsets.size(0) - 1;
        const int32_t B = total_B / T;
        TORCH_CHECK_GE(B, 0);
        TORCH_CHECK_GT(D, 0);
        TORCH_CHECK_EQ(D % 4, 0);

        Tensor output;
        SparseType o_dtype = static_cast<SparseType>(output_dtype);
        TORCH_CHECK(o_dtype == SparseType::FP32 || o_dtype == SparseType::FP16 ||
                    o_dtype == SparseType::BF16);

        int64_t adjusted_D = D;

        {%- if dense %}
        // Match output dtype to dev_weights dtype for consistency
        {%- endif %}
        output = at::empty({total_L, adjusted_D}, dev_weights.options().dtype(getScalarType(o_dtype))); // if nobag

        if (B == 0) {
            return output;
        }
        {%- if not dense %}

        {%- endif %}

        AT_DISPATCH_INDEX_TYPES(indices.scalar_type(), "batched_embedding_nobag_forward_kernel_1", [&] {
        DISPATCH_EMB_CACHE_OUTPUT_TYPES(
            dev_weights.scalar_type(),
            {%- if dense %}
            dev_weights.scalar_type(),
            {%- else %}
            lxu_cache_weights.scalar_type(),
            {%- endif %}
            output.scalar_type(),
            "batched_embedding_nobag_forward_kernel_2", [&] {
            try {
                sycl::queue& queue = c10::xpu::getCurrentXPUStream().queue();
                {%- if not dense %}
                bool use_lxu_cache = lxu_cache_weights.numel() > 0;
                {%- endif %}
                bool launched_small_kernel = false;

                DISPATCH_OPTIMAL_NOBAG_FORWARD_KERNEL(D, [&] {
                    launched_small_kernel = true;
                    constexpr size_t kSmallThreadGroupSize = kEmbeddingSize / 4;
                    constexpr size_t sg_size = kThreadGroupSize;
                    constexpr size_t kBlockDimY = kForwardMaxThreads / sg_size;
                    const size_t grid_x = {%- if dense %}(total_B + kBlockDimY - 1) / kBlockDimY{%- else %}div_round_up(static_cast<size_t>(total_B), kBlockDimY){%- endif %};

                    queue.submit([&](sycl::handler& cgh) {
                        cgh.parallel_for<{{ mdesc | capitalize }}EmbeddingNobagCodegenForwardUnweightedSmallKernel<emb_t, cache_t, output_t, index_t, kSmallThreadGroupSize>>(
                            sycl::nd_range<2>(
                                sycl::range<2>(grid_x * kBlockDimY, sg_size),
                                sycl::range<2>(kBlockDimY, sg_size)
                            ),
                            {{ mdesc | capitalize }}EmbeddingNobagCodegenForwardUnweightedSmallKernel<emb_t, cache_t, output_t, index_t, kSmallThreadGroupSize>(
                                dev_weights.packed_accessor64<emb_t, 1, RestrictPtrTraits>(),
                                {%- if not dense %}
                                uvm_weights.packed_accessor64<emb_t, 1, RestrictPtrTraits>(),
                                lxu_cache_weights.packed_accessor64<cache_t, 2, RestrictPtrTraits>(),
                                weights_placements.packed_accessor32<int32_t, 1, RestrictPtrTraits>(),
                                {%- endif %}
                                weights_offsets.packed_accessor32<int64_t, 1, RestrictPtrTraits>(),
                                D,
                                FixedDivisor(B),
                                indices.packed_accessor32<index_t, 1, RestrictPtrTraits>(),
                                offsets.packed_accessor32<index_t, 1, RestrictPtrTraits>(),
                                {%- if not dense %}
                                lxu_cache_locations.packed_accessor32<int32_t, 1, RestrictPtrTraits>(),
                                {%- endif %}
                                output.packed_accessor64<output_t, 2, RestrictPtrTraits>()
                            )
                        );
                    });

                    return;
                });

                if (!launched_small_kernel) {
                    {%- if not dense %}
                    DISPATCH_KERNEL_FOR_CACHE_CASE(use_lxu_cache, [&] {
                        {%- endif %}
                        const size_t local_x = kThreadGroupSize;{%- if dense %} {%- endif %}

                        const size_t local_y = kForwardMaxThreads / kThreadGroupSize;
                        const size_t grid{%- if not dense %}_x{%- endif %} = div_round_up(static_cast<size_t>(total_B), local_y);
                        {%- if dense %}

                        {%- else %}

                        {%- endif %}
                        queue.submit([&](sycl::handler& cgh) {
                            cgh.parallel_for<{{ mdesc | capitalize }}EmbeddingNobagCodegenForwardUnweightedKernel<emb_t, cache_t, output_t, {%- if not dense %}use_cache_t, {%- endif %}index_t, kThreadGroupSize>>(
                                sycl::nd_range<2>(
                                    sycl::range<2>(grid{%- if not dense %}_x{%- endif %} * local_y, local_x),
                                    sycl::range<2>(local_y, local_x)
                                ),
                                {{ mdesc | capitalize }}EmbeddingNobagCodegenForwardUnweightedKernel<emb_t, cache_t, output_t, {%- if not dense %}use_cache_t, {%- endif %}index_t, kThreadGroupSize>(
                                    dev_weights.packed_accessor64<emb_t, 1, RestrictPtrTraits>(),
                                    {%- if not dense %}
                                    uvm_weights.packed_accessor64<emb_t, 1, RestrictPtrTraits>(),
                                    lxu_cache_weights.packed_accessor64<cache_t, 2, RestrictPtrTraits>(),
                                    weights_placements.packed_accessor32<int32_t, 1, RestrictPtrTraits>(),
                                    {%- endif %}
                                    weights_offsets.packed_accessor32<int64_t, 1, RestrictPtrTraits>(),
                                    D,
                                    FixedDivisor(B),
                                    indices.packed_accessor32<index_t, 1, RestrictPtrTraits>(),
                                    offsets.packed_accessor32<index_t, 1, RestrictPtrTraits>(),
                                    {%- if not dense %}
                                    lxu_cache_locations.packed_accessor32<int32_t, 1, RestrictPtrTraits>(),
                                    uvm_cache_stats.size(0) == 0
                                        ? nullptr
                                        : (uvm_cache_stats.data_ptr<int32_t>() + uvm_cache_stats_index::num_conflict_unique_misses),
                                    {%- endif %}
                                    output.packed_accessor64<output_t, 2, RestrictPtrTraits>()
                                )
                            );
                        });
                    {%- if not dense %}
                    });
                    {%- endif %}
                }

            } catch (const sycl::exception& e) {
                std::cerr << "SYCL exception: " << e.what() << std::endl;
                throw;
            }
            });
        });
        return output;
    }

} // namespace fbgemm_xpu

TORCH_LIBRARY_IMPL(fbgemm, XPU, m) {
    m.impl("{{ mdesc }}_embedding_nobag_forward_unweighted_xpu", &fbgemm_xpu::{{ mdesc }}_embedding_nobag_forward_unweighted_xpu);
}
