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

{%- set optimizer = "Dense" if dense else "RowwiseAdagrad" %}

////////////////////////////////////////////////////////////////////////////////
// SYCL PORT MAPPING TO FBGEMM CUDA SOURCE - BACKWARD KERNEL TEMPLATES
////////////////////////////////////////////////////////////////////////////////
//
// This file contains the SYCL ports of the FBGEMM {{ "dense" if dense else "split" }} embedding
// backward kernels (no-bag / sequence, unweighted) with {{ "no" if dense else optimizer }} optimizer.
// It provides both a warp-per-row and a CTA-per-row kernel, dispatched by the
// host function based on segment length, plus the per-kernel helper functions.
//
// ORIGINAL CUDA SOURCES:
//   Warp kernel template:  fbgemm_gpu/codegen/training/backward/embedding_backward_split_kernel_warp_template.cu
//   Warp kernel generated: fbgemm_gpu/_skbuild/linux-x86_64-3.10/cmake-build/gen_embedding_backward_{{ "dense" if dense else "rowwise_adagrad" }}_split_unweighted_nobag_kernel_warp.cu
//   CTA kernel template:   fbgemm_gpu/codegen/training/backward/embedding_backward_split_kernel_cta_template.cu
//   CTA kernel generated:  fbgemm_gpu/_skbuild/linux-x86_64-3.10/cmake-build/gen_embedding_backward_{{ "dense" if dense else "rowwise_adagrad" }}_split_unweighted_nobag_kernel_cta.cu
//
// KERNEL MAPPING:
//   SplitEmbeddingNobagBackwardCodegen{{ optimizer }}UnweightedKernelWarpPerRow
//     → split_embedding_nobag_backward_codegen_{{ "dense" if dense else "rowwise_adagrad" }}_unweighted_kernel_warp_per_row_1 (CUDA)
//     Used for short segments (SL < max_segment_length_per_warp)
//
//   SplitEmbeddingNobagBackwardCodegen{{ optimizer }}UnweightedKernelCtaPerRow
//     → split_embedding_nobag_backward_codegen_{{ "dense" if dense else "rowwise_adagrad" }}_unweighted_kernel_cta_per_row_1 (CUDA)
//     Used for long segments (SL >= max_segment_length_per_warp)
//
// HELPER FUNCTION MAPPING:
{%- if dense %}
//   store_grad_sum  (dense only)
//     Stores the accumulated gradient for one embedding row into grad_dev_weights.
{%- else %}
//   split_rowwise_adagrad_table_update_kernel  (split only)
//     Applies the rowwise Adagrad optimizer update for one embedding row.
{%- endif %}
//
////////////////////////////////////////////////////////////////////////////////

#pragma once

#include <cassert>
#include <cstdlib>

#include <sycl/sycl.hpp>
#include <c10/xpu/XPUStream.h>

#include <ATen/ATen.h>
#include <ATen/Operators.h>
#include <ATen/core/TensorAccessor.h>
#include <ATen/native/StridedRandomAccessor.h>

{%- if not dense %}

#include <ATen/xpu/XPUGeneratorImpl.h>

{%- endif %}

#include "../fbgemm_utils/backward_utils.h"
#include "../fbgemm_utils/tensor_utils.h"
#include "../fbgemm_utils/utils.h"

{%- if not dense %}

#include "../fbgemm_utils/split_embeddings_cache_xpu.h"
#include "../fbgemm_utils/weight_row.h"

{%- endif %}


using Tensor = at::Tensor;
using at::native::RestrictPtrTraits;
{%- if not dense %}
using namespace fbgemm_xpu::utils;
using float4 = sycl::float4;
{%- endif %}

namespace fbgemm_xpu {

{%- if dense %}


    ////////////////////////////////////////////////////////////////////////////////
    // store_grad_sum - Gradient Storage Helper (Dense Only)
    ////////////////////////////////////////////////////////////////////////////////
    //
    // DESCRIPTION:
    //   Stores the accumulated per-row gradient vector (grad_sum) from a backward
    //   kernel into the grad_dev_weights output tensor at position
    //   [weights_offset + idx * D].
    //
    //   Two storage paths controlled by kUseVecBlocking:
    //     kUseVecBlocking = true  — max_vecs is a runtime value; reads from
    //                               smem_grad_sum (shared memory).
    //     kUseVecBlocking = false — kFixedMaxVecsPerThread is a compile-time
    //                               constant; reads directly from grad_sum
    //                               registers, enabling loop unrolling.
    //
    ////////////////////////////////////////////////////////////////////////////////

    template<
        typename emb_t,
        typename cache_t,
        int32_t kFixedMaxVecsPerThread,
        int32_t kThreadGroupSize,
        int32_t VEC_WIDTH,
        bool kUseVecBlocking
    >
    void store_grad_sum(
        const sycl::nd_item<2>& item,
        at::PackedTensorAccessor64<emb_t, 1, RestrictPtrTraits>& grad_dev_weights,
        const Vec4TAcc<cache_t>* grad_sum,
        const Vec4TAcc<cache_t>* smem_grad_sum,
        const int32_t D,
        const int64_t weights_offset,
        const int64_t idx,
        const int32_t max_vecs_per_thread
    ) {
        const auto threadIdx_x = item.get_local_id(1);

        // Copy value to max_vecs to make max_vecs_per_thread known at compile time
        // when kUseVecBlocking == false
        const int32_t max_vecs =
            kUseVecBlocking ? max_vecs_per_thread : kFixedMaxVecsPerThread;

        if constexpr (kUseVecBlocking) {
            // max_vecs is not known at compile time
            for (int32_t vec = 0;
                vec < max_vecs &&
                (kThreadGroupSize * vec + threadIdx_x) * VEC_WIDTH < D;
                ++vec) {
                const int32_t d_vec = vec * kThreadGroupSize + threadIdx_x;
                [[maybe_unused]] const int32_t d = d_vec * VEC_WIDTH;

                auto& grad = smem_grad_sum[d_vec];
                grad.store(&grad_dev_weights[weights_offset + idx * D + d]);
            }
        } else {
            // kFixedMaxVecsPerThread is known at compile time
            #pragma unroll kFixedMaxVecsPerThread
            for (int32_t vec = 0;
                vec < kFixedMaxVecsPerThread
                    && (kThreadGroupSize * vec + threadIdx_x) * VEC_WIDTH < D;
                ++vec) {
                const int32_t d_vec = vec * kThreadGroupSize + threadIdx_x;
                [[maybe_unused]] const int32_t d = d_vec * VEC_WIDTH;

                auto& grad = grad_sum[vec];
                grad.store(&grad_dev_weights[weights_offset + idx * D + d]);
            }
        }
    }

{%- else %}


    ////////////////////////////////////////////////////////////////////////////////
    // split_rowwise_adagrad_table_update_kernel - Rowwise Adagrad Update Helper
    ////////////////////////////////////////////////////////////////////////////////
    //
    // DESCRIPTION:
    //   Applies the rowwise Adagrad optimizer update for a single embedding row
    //   after its gradient (grad_sum) has been accumulated by the backward kernel.
    //
    //   Steps performed per call:
    //     1. Resolve weight pointer from PlacementType (DEVICE / UVM / MANAGED_CACHING).
    //     2. Optionally read the cached row from lxu_cache_weights when present.
    //     3. Compute the per-row gradient L2 norm (g_avg_square) using sub-group reduce.
    //     4. Update momentum1 (running sum of squared gradients) on thread 0 and
    //        broadcast multiplier and correction factors to the sub-group.
    //     5. Apply weight update: w = correction * w - multiplier * grad.
    //     6. Optionally clip the updated weight row to max_norm.
    //
    //   Supports L2 regularization (weight_decay_mode == 1) and
    //   decoupled weight decay (weight_decay_mode == 2 / 5).
    //   Two accumulation paths (kUseVecBlocking) mirror store_grad_sum above.
    //
    ////////////////////////////////////////////////////////////////////////////////

    template <
        typename emb_t,
        typename cache_t,
        int32_t kFixedMaxVecsPerThread,
        int32_t kThreadGroupSize,
        int32_t VEC_WIDTH,
        bool kUseVecBlocking
    >
    void split_rowwise_adagrad_table_update_kernel(
        at::PackedTensorAccessor64<emb_t, 1, RestrictPtrTraits>& dev_weights,
        at::PackedTensorAccessor64<emb_t, 1, RestrictPtrTraits>& uvm_weights,
        at::PackedTensorAccessor64<cache_t, 2, RestrictPtrTraits>& lxu_cache_weights,
        const at::PackedTensorAccessor32<int32_t, 1, RestrictPtrTraits>& weights_placements,
        const at::PackedTensorAccessor32<int64_t, 1, RestrictPtrTraits>& weights_offsets,
        const at::PackedTensorAccessor32<int32_t, 1, RestrictPtrTraits>& sorted_lxu_cache_locations,
        Vec4TAcc<cache_t>* grad_sum,
        Vec4TAcc<cache_t>* smem_grad_sum,
        Vec4TAcc<cache_t>* shared_weight_update_row,
        const bool stochastic_rounding,
        const PhiloxXpuState& stochastic_rounding_philox_args,
        const uint32_t run_id,
        const uint32_t cache_loc_run_id,
        const int32_t D,
        const int32_t t,
        const int64_t idx,
        const float global_weight_decay,
        const int32_t max_vecs_per_thread,
        at::PackedTensorAccessor64<at::acc_type<cache_t, true>, 1, RestrictPtrTraits>& momentum1_dev,
        at::PackedTensorAccessor64<at::acc_type<cache_t, true>, 1, RestrictPtrTraits>& momentum1_uvm,
        at::PackedTensorAccessor32<int32_t, 1, RestrictPtrTraits>& momentum1_placements,
        at::PackedTensorAccessor32<int64_t, 1, RestrictPtrTraits>& momentum1_offsets,
        const sycl::nd_item<2>& item,
        float learning_rate = 0,
        float eps = 0,
        float weight_decay = 0.0,
        int64_t weight_decay_mode = 0,
        float max_norm = 0.0
    ) {
        using acc_t = at::acc_type<cache_t, true>;
        const auto threadIdx_x = item.get_local_id(1);
        const auto blockDim_x = item.get_local_range(1);
        const auto sg = item.get_sub_group();

        const int32_t max_vecs =
            kUseVecBlocking ? max_vecs_per_thread : kFixedMaxVecsPerThread;

        // ========== Setup weight pointers based on placement ==========
        const int64_t weights_offset = weights_offsets[t];
        emb_t* __restrict__ weights {nullptr};
        cache_t* __restrict__ cache_weights {nullptr};
        int32_t D_emb = D;

        const auto weights_placement = static_cast<PlacementType>(weights_placements[t]);
        if (weights_placement == PlacementType::DEVICE) {
            weights = &dev_weights[weights_offset + idx * D_emb];
        } else {
            weights = &uvm_weights[weights_offset + idx * D_emb];
        }

        if (weights_placement == PlacementType::MANAGED_CACHING) {
            const auto cache_idx = sorted_lxu_cache_locations[cache_loc_run_id];
            if (cache_idx != kCacheLocationMissing) {
                cache_weights = &lxu_cache_weights[cache_idx][0];
            }
        }

        // ========== Setup momentum pointer based on placement ==========
        acc_t* __restrict__ momentum1;
        const auto momentum1_placement = static_cast<PlacementType>(momentum1_placements[t]);
        const int64_t momentum1_offset = momentum1_offsets[t];
        if (momentum1_placement == PlacementType::DEVICE) {
            momentum1 = &momentum1_dev[momentum1_offset];
        } else {
            momentum1 = &momentum1_uvm[momentum1_offset];
        }

        // ========== Create weight row accessor ==========
        auto weight_row_template =
            WeightRow<emb_t, cache_t, acc_t>(
                weights,
                cache_weights,
                D,
                stochastic_rounding,
                &stochastic_rounding_philox_args,
                threadIdx_x + run_id * blockDim_x);

        // ========== Compute gradient norm with weight decay ==========
        acc_t g_local_sum_square = 0.0;

        if constexpr (kUseVecBlocking) {
            for (int32_t vec = 0;
                vec < max_vecs &&
                (kThreadGroupSize * vec + threadIdx_x) * VEC_WIDTH < D;
                ++vec) {
                const int32_t d_vec = vec * kThreadGroupSize + threadIdx_x;
                const int32_t d = d_vec * VEC_WIDTH;

                const float4* grad = &smem_grad_sum[d_vec].acc;
                auto gx = grad->x();
                auto gy = grad->y();
                auto gz = grad->z();
                auto gw = grad->w();

                if (weight_decay_mode == 1) {
                    // L2 regularization
                    Vec4TAcc<cache_t> weight = weight_row_template.load(d);
                    gx += weight_decay * weight.acc.x();
                    gy += weight_decay * weight.acc.y();
                    gz += weight_decay * weight.acc.z();
                    gw += weight_decay * weight.acc.w();
                }
                g_local_sum_square += gx * gx + gy * gy + gz * gz + gw * gw;
            }
        } else {
            #pragma unroll kFixedMaxVecsPerThread
            for (int32_t vec = 0;
                vec < kFixedMaxVecsPerThread
                    && (kThreadGroupSize * vec + threadIdx_x) * VEC_WIDTH < D;
                ++vec) {
                const int32_t d_vec = vec * kThreadGroupSize + threadIdx_x;
                const int32_t d = d_vec * VEC_WIDTH;

                const float4* grad = &grad_sum[vec].acc;
                auto gx = grad->x();
                auto gy = grad->y();
                auto gz = grad->z();
                auto gw = grad->w();

                if (weight_decay_mode == 1) {
                    // L2 regularization
                    Vec4TAcc<cache_t> weight = weight_row_template.load(d);
                    gx += weight_decay * weight.acc.x();
                    gy += weight_decay * weight.acc.y();
                    gz += weight_decay * weight.acc.z();
                    gw += weight_decay * weight.acc.w();
                }
                g_local_sum_square += gx * gx + gy * gy + gz * gz + gw * gw;
            }
        }

        // ========== Update optimizer state (Rowwise Adagrad) ==========
        acc_t group_sum = sycl::reduce_over_group(sg, g_local_sum_square, sycl::plus<acc_t>{});
        const acc_t g_avg_square = group_sum / static_cast<acc_t>(D);

        acc_t multiplier = 0.0;
        acc_t correction = 0.0;

        if (threadIdx_x == 0) {
            acc_t new_sum_square_grads = g_avg_square + momentum1[idx];
            momentum1[idx] = new_sum_square_grads;

            multiplier = learning_rate / (sqrtf(new_sum_square_grads) + eps);

            if (weight_decay_mode == 1) {
                // L2 regularization
                correction = 1.0 - multiplier * weight_decay;
            } else if (weight_decay_mode == 2 || weight_decay_mode == 5) {
                // Decoupled weight decay
                correction = 1.0 - learning_rate * weight_decay;
            } else {
                correction = 1.0;
            }
        }

        multiplier = sycl::group_broadcast(sg, multiplier, 0);
        correction = sycl::group_broadcast(sg, correction, 0);

        // ========== Apply weight update ==========
        if constexpr (kUseVecBlocking) {
            for (int32_t vec = 0;
                vec < max_vecs &&
                (kThreadGroupSize * vec + threadIdx_x) * VEC_WIDTH < D;
                ++vec) {
                const int32_t d_vec = vec * kThreadGroupSize + threadIdx_x;
                const int32_t d = d_vec * VEC_WIDTH;

                Vec4TAcc<cache_t> weight_new = weight_row_template.load(d);
                Vec4TAcc<cache_t>& grad = smem_grad_sum[d_vec];
                weight_new.mul_(global_weight_decay);

                weight_new.acc.x() = correction * weight_new.acc.x() - multiplier * grad.acc.x();
                weight_new.acc.y() = correction * weight_new.acc.y() - multiplier * grad.acc.y();
                weight_new.acc.z() = correction * weight_new.acc.z() - multiplier * grad.acc.z();
                weight_new.acc.w() = correction * weight_new.acc.w() - multiplier * grad.acc.w();

                weight_row_template.store(weight_new, d);
            }
        } else {
            #pragma unroll kFixedMaxVecsPerThread
            for (int32_t vec = 0;
                vec < kFixedMaxVecsPerThread
                    && (kThreadGroupSize * vec + threadIdx_x) * VEC_WIDTH < D;
                ++vec) {
                const int32_t d_vec = vec * kThreadGroupSize + threadIdx_x;
                const int32_t d = d_vec * VEC_WIDTH;

                Vec4TAcc<cache_t> weight_new = weight_row_template.load(d);
                Vec4TAcc<cache_t>& grad = grad_sum[vec];
                weight_new.mul_(global_weight_decay);

                weight_new.acc.x() = correction * weight_new.acc.x() - multiplier * grad.acc.x();
                weight_new.acc.y() = correction * weight_new.acc.y() - multiplier * grad.acc.y();
                weight_new.acc.z() = correction * weight_new.acc.z() - multiplier * grad.acc.z();
                weight_new.acc.w() = correction * weight_new.acc.w() - multiplier * grad.acc.w();

                weight_row_template.store(weight_new, d);
            }
        }

        // ========== Apply max_norm constraint if specified ==========
        if (max_norm > 0.0) {
            assert(!(std::is_same_v<emb_t, uint8_t> && !cache_weights)
                && "max_norm not supported for uint8 without cache");

            // Compute weight norm
            acc_t weight_sum_square = 0.0;
            for (int32_t vec = 0;
                vec < max_vecs && (kThreadGroupSize * vec + threadIdx_x) * VEC_WIDTH < D;
                ++vec) {
                const int32_t d = (kThreadGroupSize * vec + threadIdx_x) * VEC_WIDTH;
                Vec4TAcc<cache_t> weight_new = weight_row_template.load(d);
                weight_sum_square
                    += weight_new.acc.x() * weight_new.acc.x()
                    + weight_new.acc.y() * weight_new.acc.y()
                    + weight_new.acc.z() * weight_new.acc.z()
                    + weight_new.acc.w() * weight_new.acc.w();
            }

            acc_t weight_norm = sycl::sqrt(
                sycl::reduce_over_group(sg, weight_sum_square, sycl::plus<acc_t>{}));

            // Clip gradient if norm exceeds max_norm
            acc_t clip_factor = 1.0;
            if (threadIdx_x == 0 && weight_norm > max_norm) {
                clip_factor = max_norm / (weight_norm + 1e-8);
            }
            clip_factor = sycl::group_broadcast(sg, clip_factor, 0);

            if (clip_factor < 1.0) {
                if constexpr (kUseVecBlocking) {
                    for (int32_t vec = 0;
                        vec < max_vecs &&
                        (kThreadGroupSize * vec + threadIdx_x) * VEC_WIDTH < D;
                        ++vec) {
                        const int32_t d = (kThreadGroupSize * vec + threadIdx_x) * VEC_WIDTH;
                        Vec4TAcc<cache_t> weight_new = weight_row_template.load(d);
                        weight_new.mul_(clip_factor);
                        weight_row_template.store(weight_new, d);
                    }
                } else {
                    #pragma unroll kFixedMaxVecsPerThread
                    for (int32_t vec = 0;
                        vec < kFixedMaxVecsPerThread
                            && (kThreadGroupSize * vec + threadIdx_x) * VEC_WIDTH < D;
                        ++vec) {
                        const int32_t d = (kThreadGroupSize * vec + threadIdx_x) * VEC_WIDTH;
                        Vec4TAcc<cache_t> weight_new = weight_row_template.load(d);
                        weight_new.mul_(clip_factor);
                        weight_row_template.store(weight_new, d);
                    }
                }
            }
        }
    }

{%- endif %}



    ////////////////////////////////////////////////////////////////////////////////
    // SplitEmbeddingNobagBackwardCodegen{{ optimizer }}UnweightedKernelWarpPerRow - Warp-Per-Row Backward Kernel
    ////////////////////////////////////////////////////////////////////////////////
    //
    // CUDA SOURCE MAPPING:
    //   CUDA Kernel:   split_embedding_nobag_backward_codegen_{{ "dense" if dense else "rowwise_adagrad" }}_unweighted_kernel_warp_per_row_1
    //   CUDA File:     fbgemm_gpu/_skbuild/linux-x86_64-3.10/cmake-build/gen_embedding_backward_{{ "dense" if dense else "rowwise_adagrad" }}_split_unweighted_nobag_kernel_warp.cu
    //   CUDA Template: fbgemm_gpu/codegen/training/backward/embedding_backward_split_kernel_warp_template.cu
    //
    // DESCRIPTION:
    //   Backward kernel for short segments (SL < max_segment_length_per_warp).
    //   One sub-group (warp) handles the full gradient accumulation and weight
    //   update for a single embedding row. Uses grid-striding over run IDs so
    //   each work-item processes multiple rows when the grid is smaller than the
    //   number of unique indices.
{%- if dense %}
    //   Dense path: accumulates gradients and stores to grad_dev_weights (no optimizer).
{%- else %}
    //   Split path: accumulates gradients then calls split_rowwise_adagrad_table_update_kernel.
{%- endif %}
    //
    ////////////////////////////////////////////////////////////////////////////////

    template <
        typename emb_t,
        typename grad_t,
        typename cache_t,
        typename index_t,
        int32_t kFixedMaxVecsPerThread,
        int32_t kThreadGroupSize,
        bool kUseVecBlocking>
    class SplitEmbeddingNobagBackwardCodegen{{ optimizer }}UnweightedKernelWarpPerRow {
    public:
        SplitEmbeddingNobagBackwardCodegen{{ optimizer }}UnweightedKernelWarpPerRow(
            const at::PackedTensorAccessor64<grad_t, 2, RestrictPtrTraits> grad_output,
            at::PackedTensorAccessor64<emb_t, 1, RestrictPtrTraits> dev_weights,
{%- if not dense %}
            at::PackedTensorAccessor64<emb_t, 1, RestrictPtrTraits> uvm_weights,
            at::PackedTensorAccessor64<cache_t, 2, RestrictPtrTraits> lxu_cache_weights,
            const at::PackedTensorAccessor32<int32_t, 1, RestrictPtrTraits> weights_placements,
{%- endif %}
            const at::PackedTensorAccessor32<int64_t, 1, RestrictPtrTraits> weights_offsets,
            int64_t D,
            const at::PackedTensorAccessor32<int64_t, 1, RestrictPtrTraits> hash_size_cumsum,
            const at::PackedTensorAccessor32<index_t, 1, RestrictPtrTraits> sorted_linear_indices_run,
            const at::PackedTensorAccessor32<int32_t, 1, RestrictPtrTraits> sorted_linear_indices_cumulative_run_lengths,
            const at::PackedTensorAccessor32<int64_t, 1, RestrictPtrTraits> sorted_infos,
{%- if not dense %}
            const at::PackedTensorAccessor32<int32_t, 1, RestrictPtrTraits> sorted_lxu_cache_locations,
            const bool use_uniq_cache_locations,
            const at::PackedTensorAccessor32<int32_t, 1, RestrictPtrTraits> table_unique_indices_offsets,
{%- endif %}
            const at::PackedTensorAccessor32<int32_t, 1, RestrictPtrTraits> sorted_linear_indices_num_runs,
            int32_t max_segment_length_per_warp,
{%- if not dense %}
            bool stochastic_rounding,
            PhiloxXpuState stochastic_rounding_philox_args,
{%- else %}
            at::PackedTensorAccessor64<emb_t, 1, RestrictPtrTraits> grad_dev_weights,
{%- endif %}
            const int32_t max_D,
            const int32_t max_vecs_per_thread,
{%- if not dense %}
            at::PackedTensorAccessor64<at::acc_type<cache_t, true>, 1, RestrictPtrTraits> momentum1_dev,
            at::PackedTensorAccessor64<at::acc_type<cache_t, true>, 1, RestrictPtrTraits> momentum1_uvm,
            at::PackedTensorAccessor32<int32_t, 1, RestrictPtrTraits> momentum1_placements,
            at::PackedTensorAccessor32<int64_t, 1, RestrictPtrTraits> momentum1_offsets,
{%- endif %}
            sycl::local_accessor<cache_t, 1> smem,
{%- if dense %}
            double unused = 0
{%- else %}
            float learning_rate = 0,
            float eps = 0,
            float weight_decay = 0.0,
            int64_t weight_decay_mode = 0,
            float max_norm = 0.0
{%- endif %}
        ) : grad_output_(grad_output),
            dev_weights_(dev_weights),
{%- if not dense %}
            uvm_weights_(uvm_weights),
            lxu_cache_weights_(lxu_cache_weights),
            weights_placements_(weights_placements),
{%- endif %}
            weights_offsets_(weights_offsets),
            D_(D),
            hash_size_cumsum_(hash_size_cumsum),
            sorted_linear_indices_run_(sorted_linear_indices_run),
            sorted_linear_indices_cumulative_run_lengths_(sorted_linear_indices_cumulative_run_lengths),
            sorted_infos_(sorted_infos),
{%- if not dense %}
            sorted_lxu_cache_locations_(sorted_lxu_cache_locations),
            use_uniq_cache_locations_(use_uniq_cache_locations),
            table_unique_indices_offsets_(table_unique_indices_offsets),
{%- endif %}
            sorted_linear_indices_num_runs_(sorted_linear_indices_num_runs),
            max_segment_length_per_warp_(max_segment_length_per_warp),
{%- if not dense %}
            stochastic_rounding_(stochastic_rounding),
            stochastic_rounding_philox_args_(stochastic_rounding_philox_args),
{%- endif %}
            max_D_(max_D),
            max_vecs_per_thread_(max_vecs_per_thread),
{%- if not dense %}
            momentum1_dev_(momentum1_dev),
            momentum1_uvm_(momentum1_uvm),
            momentum1_placements_(momentum1_placements),
            momentum1_offsets_(momentum1_offsets),
{%- else %}
            grad_dev_weights_(grad_dev_weights),
{%- endif %}
            smem_(smem)
{%- if not dense %},
            learning_rate_(learning_rate),
            eps_(eps),
            weight_decay_(weight_decay),
            weight_decay_mode_(weight_decay_mode),
            max_norm_(max_norm)
{%- endif %}
            {}

        void SYCL_EXTERNAL operator()(const sycl::nd_item<2>& item) const {

        constexpr int VEC_WIDTH = 4;
        const auto threadIdx_x = item.get_local_id(1);
        const auto threadIdx_y = item.get_local_id(0);
        const auto blockIdx_x = item.get_group(0);
        const auto gridDim_x = item.get_group_range(0);
        const auto blockDim_y = item.get_local_range(0);
        const auto sg = item.get_sub_group();

        const int32_t T = weights_offsets_.size(0);
        const auto start_run_id = blockIdx_x * blockDim_y + threadIdx_y;

        // Setup shared memory for gradient accumulation
        const int32_t grad_sum_stride = max_D_ / VEC_WIDTH;
        auto* smem_grad_sum = kUseVecBlocking
            ? reinterpret_cast<Vec4TAcc<cache_t>*>(smem_.get_pointer().get())
                + threadIdx_y * grad_sum_stride
            : nullptr;

        // Iterate over all runs (unique indices) assigned to this work-item
        for (uint32_t run_id = start_run_id;
             run_id < sorted_linear_indices_run_.size(0) && run_id < sorted_linear_indices_num_runs_[0];
             run_id += gridDim_x * blockDim_y) {

            const int64_t linear_index = sorted_linear_indices_run_[run_id];
            const int32_t segment_start = sorted_linear_indices_cumulative_run_lengths_[run_id];
            const int32_t segment_end = sorted_linear_indices_cumulative_run_lengths_[run_id + 1];
            const int32_t SL = segment_end - segment_start;

            // Skip long segments (handled by CTA-per-row kernel)
            if (SL >= max_segment_length_per_warp_) {
                continue;
            }

            // Decode table ID from sorted info
            const auto info_0 = sorted_infos_[segment_start];
            int32_t t_0 = info_0 % T;

            // Calculate embedding index
            int64_t hash_size = hash_size_cumsum_[t_0];
            int64_t idx = linear_index - hash_size;

            // Gradient accumulation buffer
            Vec4TAcc<cache_t> grad_sum[kFixedMaxVecsPerThread];
            constexpr int32_t kGroupVecWidth = kThreadGroupSize * VEC_WIDTH;
            const int32_t num_vecs = (D_ + kGroupVecWidth - 1) / kGroupVecWidth;

            // Compute gradient sum across all occurrences of this index
            compute_grad_sum_unweighted_nobag<
                grad_t,
                cache_t,
                kFixedMaxVecsPerThread,
                kThreadGroupSize,
                VEC_WIDTH,
                kUseVecBlocking>(
                    item,
                    grad_sum,
                    smem_grad_sum,
                    grad_output_,
                    static_cast<int32_t>(D_),
                    T,
                    sorted_infos_,
                    segment_start,
                    0,
                    SL,
                    num_vecs
            );

            const int32_t max_vecs =
                kUseVecBlocking ? max_vecs_per_thread_ : kFixedMaxVecsPerThread;

{%- if dense %}

            // ========== DENSE: Store gradient to grad_dev_weights ==========
            const int64_t weights_offset = weights_offsets_[t_0];
            store_grad_sum<
                emb_t,
                cache_t,
                kFixedMaxVecsPerThread,
                kThreadGroupSize,
                VEC_WIDTH,
                kUseVecBlocking>(
                    item,
                    grad_dev_weights_,
                    grad_sum,
                    kUseVecBlocking ? smem_grad_sum : nullptr,
                    static_cast<int32_t>(D_),
                    weights_offset,
                    idx,
                    max_vecs
            );

{%- else %}

            // ========== SPLIT: Apply optimizer (Rowwise Adagrad) ==========
            split_rowwise_adagrad_table_update_kernel<
                emb_t,
                cache_t,
                kFixedMaxVecsPerThread,
                kThreadGroupSize,
                VEC_WIDTH,
                kUseVecBlocking>(
                    dev_weights_,
                    uvm_weights_,
                    lxu_cache_weights_,
                    weights_placements_,
                    weights_offsets_,
                    sorted_lxu_cache_locations_,
                    grad_sum,
                    smem_grad_sum,
                    smem_grad_sum, // shared_weight_update_row (reuse smem_grad_sum)
                    stochastic_rounding_,
                    stochastic_rounding_philox_args_,
                    run_id,
                    segment_start, // cache_loc_run_id
                    D_,
                    t_0,
                    idx,
                    1.0f, // global_weight_decay
                    max_vecs,
                    momentum1_dev_,
                    momentum1_uvm_,
                    momentum1_placements_,
                    momentum1_offsets_,
                    item,
                    learning_rate_,
                    eps_,
                    weight_decay_,
                    weight_decay_mode_,
                    max_norm_
            ); // if not dense and optimizer != "none"

{%- endif %}

        } // for each run
        }

    private:
        const at::PackedTensorAccessor64<grad_t, 2, RestrictPtrTraits> grad_output_;
        mutable at::PackedTensorAccessor64<emb_t, 1, RestrictPtrTraits> dev_weights_;
{%- if not dense %}
        mutable at::PackedTensorAccessor64<emb_t, 1, RestrictPtrTraits> uvm_weights_;
        mutable at::PackedTensorAccessor64<cache_t, 2, RestrictPtrTraits> lxu_cache_weights_;
        const at::PackedTensorAccessor32<int32_t, 1, RestrictPtrTraits> weights_placements_;
{%- endif %}
        const at::PackedTensorAccessor32<int64_t, 1, RestrictPtrTraits> weights_offsets_;
        int64_t D_;
        const at::PackedTensorAccessor32<int64_t, 1, RestrictPtrTraits> hash_size_cumsum_;
        const at::PackedTensorAccessor32<index_t, 1, RestrictPtrTraits> sorted_linear_indices_run_;
        const at::PackedTensorAccessor32<int32_t, 1, RestrictPtrTraits> sorted_linear_indices_cumulative_run_lengths_;
        const at::PackedTensorAccessor32<int64_t, 1, RestrictPtrTraits> sorted_infos_;
{%- if not dense %}
        const at::PackedTensorAccessor32<int32_t, 1, RestrictPtrTraits> sorted_lxu_cache_locations_;
        const bool use_uniq_cache_locations_;
        const at::PackedTensorAccessor32<int32_t, 1, RestrictPtrTraits> table_unique_indices_offsets_;
{%- endif %}
        const at::PackedTensorAccessor32<int32_t, 1, RestrictPtrTraits> sorted_linear_indices_num_runs_;
        int32_t max_segment_length_per_warp_;
{%- if not dense %}
        bool stochastic_rounding_;
        PhiloxXpuState stochastic_rounding_philox_args_;
{%- endif %}
        int32_t max_D_;
        int32_t max_vecs_per_thread_;
{%- if not dense %}
        mutable at::PackedTensorAccessor64<at::acc_type<cache_t, true>, 1, RestrictPtrTraits> momentum1_dev_;
        mutable at::PackedTensorAccessor64<at::acc_type<cache_t, true>, 1, RestrictPtrTraits> momentum1_uvm_;
        mutable at::PackedTensorAccessor32<int32_t, 1, RestrictPtrTraits> momentum1_placements_;
        mutable at::PackedTensorAccessor32<int64_t, 1, RestrictPtrTraits> momentum1_offsets_;
{%- else %}
        mutable at::PackedTensorAccessor64<emb_t, 1, RestrictPtrTraits> grad_dev_weights_;
{%- endif %}
        sycl::local_accessor<cache_t, 1> smem_;
{%- if not dense %}
        float learning_rate_;
        float eps_;
        float weight_decay_;
        int64_t weight_decay_mode_;
        float max_norm_;
{%- endif %}
    };


    ////////////////////////////////////////////////////////////////////////////////
    // SplitEmbeddingNobagBackwardCodegen{{ optimizer }}UnweightedKernelCtaPerRow - CTA-Per-Row Backward Kernel
    ////////////////////////////////////////////////////////////////////////////////
    //
    // CUDA SOURCE MAPPING:
    //   CUDA Kernel:   split_embedding_nobag_backward_codegen_{{ "dense" if dense else "codegen_rowwise_adagrad" }}_unweighted_kernel_cta_per_row_1
    //   CUDA File:     fbgemm_gpu/_skbuild/linux-x86_64-3.10/cmake-build/gen_embedding_backward_{{ "dense" if dense else "rowwise_adagrad" }}_split_unweighted_nobag_kernel_cta.cu
    //   CUDA Template: fbgemm_gpu/codegen/training/backward/embedding_backward_split_kernel_cta_template.cu
    //
    // DESCRIPTION:
    //   Backward kernel for long segments (SL >= max_segment_length_per_warp).
    //   One CTA (thread block) handles the gradient accumulation and weight update
    //   for a single embedding row. Multiple warps within the CTA each process a
    //   slice of the segment and reduce into shared memory before the final update.
    //   For very long segments that exceed max_segment_length_per_cta, multiple
    //   CTAs cooperate via temp_grad_accum and grad_accum_counter, with the last
    //   CTA to finish performing the optimizer step.
{%- if dense %}
    //   Dense path: accumulates gradients and stores to grad_dev_weights (no optimizer).
{%- else %}
    //   Split path: accumulates gradients then calls split_rowwise_adagrad_table_update_kernel.
{%- endif %}
    //
    ////////////////////////////////////////////////////////////////////////////////

    template <
        typename emb_t,
        typename grad_t,
        typename cache_t,
        typename index_t,
        int32_t kFixedMaxVecsPerThread,
        int32_t kThreadGroupSize,
        bool kUseVecBlocking>
    class SplitEmbeddingNobagBackwardCodegen{{ optimizer }}UnweightedKernelCtaPerRow {
    public:
        SplitEmbeddingNobagBackwardCodegen{{ optimizer }}UnweightedKernelCtaPerRow(
            const at::PackedTensorAccessor64<grad_t, 2, RestrictPtrTraits> grad_output,
            at::PackedTensorAccessor64<emb_t, 1, RestrictPtrTraits> dev_weights,
{%- if not dense %}
            at::PackedTensorAccessor64<emb_t, 1, RestrictPtrTraits> uvm_weights,
            at::PackedTensorAccessor64<cache_t, 2, RestrictPtrTraits> lxu_cache_weights,
            const at::PackedTensorAccessor32<int32_t, 1, RestrictPtrTraits> weights_placements,
{%- endif %}
            const at::PackedTensorAccessor32<int64_t, 1, RestrictPtrTraits> weights_offsets,
            int64_t D,
            const at::PackedTensorAccessor32<int64_t, 1, RestrictPtrTraits> hash_size_cumsum,
            const at::PackedTensorAccessor32<index_t, 1, RestrictPtrTraits> sorted_linear_indices_run,
            const at::PackedTensorAccessor32<int32_t, 1, RestrictPtrTraits> sorted_linear_indices_cumulative_run_lengths,
            const at::PackedTensorAccessor32<int32_t, 1, RestrictPtrTraits> long_run_ids,
            const at::PackedTensorAccessor32<int32_t, 1, RestrictPtrTraits> num_long_run_ids,
            const at::PackedTensorAccessor32<int64_t, 1, RestrictPtrTraits> sorted_infos,
{%- if dense %}
            at::PackedTensorAccessor64<emb_t, 1, RestrictPtrTraits> grad_dev_weights,
{%- else %}
            const at::PackedTensorAccessor32<int32_t, 1, RestrictPtrTraits> sorted_lxu_cache_locations,
            const bool use_uniq_cache_locations,
            const at::PackedTensorAccessor32<int32_t, 1, RestrictPtrTraits> table_unique_indices_offsets,
            bool stochastic_rounding,
            PhiloxXpuState stochastic_rounding_philox_args,
{%- endif %}
            const at::PackedTensorAccessor32<int32_t, 1, RestrictPtrTraits> long_run_id_to_really_long_run_ids,
            at::PackedTensorAccessor32<at::acc_type<cache_t, true>, 2, RestrictPtrTraits> temp_grad_accum,
            at::PackedTensorAccessor32<int32_t, 1, RestrictPtrTraits> grad_accum_counter,
            const int32_t max_segment_length_per_cta,
            const bool use_deterministic_algorithms,
            const int32_t max_vecs_per_thread,
{%- if not dense %}
            at::PackedTensorAccessor64<at::acc_type<cache_t, true>, 1, RestrictPtrTraits> momentum1_dev,
            at::PackedTensorAccessor64<at::acc_type<cache_t, true>, 1, RestrictPtrTraits> momentum1_uvm,
            at::PackedTensorAccessor32<int32_t, 1, RestrictPtrTraits> momentum1_placements,
            at::PackedTensorAccessor32<int64_t, 1, RestrictPtrTraits> momentum1_offsets,
{%- endif %}
            sycl::local_accessor<cache_t, 1> smem,
{%- if dense %}
            float unused = 0
{%- else %}
            float learning_rate = 0,
            float eps = 0,
            float weight_decay = 0.0,
            int64_t weight_decay_mode = 0,
            float max_norm = 0.0
{%- endif %}
        ) : grad_output_(grad_output),
            dev_weights_(dev_weights),
{%- if not dense %}
            uvm_weights_(uvm_weights),
            lxu_cache_weights_(lxu_cache_weights),
            weights_placements_(weights_placements),
{%- endif %}
            weights_offsets_(weights_offsets),
            D_(D),
            hash_size_cumsum_(hash_size_cumsum),
            sorted_linear_indices_run_(sorted_linear_indices_run),
            sorted_linear_indices_cumulative_run_lengths_(sorted_linear_indices_cumulative_run_lengths),
            long_run_ids_(long_run_ids),
            num_long_run_ids_(num_long_run_ids),
            sorted_infos_(sorted_infos),
{%- if dense %}
            grad_dev_weights_(grad_dev_weights),
{%- else %}
            sorted_lxu_cache_locations_(sorted_lxu_cache_locations),
            use_uniq_cache_locations_(use_uniq_cache_locations),
            table_unique_indices_offsets_(table_unique_indices_offsets),
            stochastic_rounding_(stochastic_rounding),
            stochastic_rounding_philox_args_(stochastic_rounding_philox_args),
{%- endif %}
            long_run_id_to_really_long_run_ids_(long_run_id_to_really_long_run_ids),
            temp_grad_accum_(temp_grad_accum),
            grad_accum_counter_(grad_accum_counter),
            max_segment_length_per_cta_(max_segment_length_per_cta),
            use_deterministic_algorithms_(use_deterministic_algorithms),
            max_vecs_per_thread_(max_vecs_per_thread),
{%- if not dense %}
            momentum1_dev_(momentum1_dev),
            momentum1_uvm_(momentum1_uvm),
            momentum1_placements_(momentum1_placements),
            momentum1_offsets_(momentum1_offsets),
{%- endif %}
            smem_(smem)
{%- if not dense %},
            learning_rate_(learning_rate),
            eps_(eps),
            weight_decay_(weight_decay),
            weight_decay_mode_(weight_decay_mode),
            max_norm_(max_norm)
{%- endif %}
            {}

        void SYCL_EXTERNAL operator()(const sycl::nd_item<2>& item) const {

        constexpr int VEC_WIDTH = 4;
{%- if not dense %}
        constexpr auto kIsInt8 = false;
{%- endif %}
        int32_t T = weights_offsets_.size(0);
        const int32_t num_long_runs = num_long_run_ids_[0];
        const auto warp_id = item.get_local_id(0);
        const auto lane_id = item.get_local_id(1);
        const auto threadIdx_x = item.get_local_id(1);
        const auto blockIdx_x = item.get_group(0);
        const auto gridDim_x = item.get_group_range(0);
        const auto blockDim_y = item.get_local_range(0);
        const auto sg = item.get_sub_group();

        const int32_t max_vecs =
            kUseVecBlocking ? max_vecs_per_thread_ : kFixedMaxVecsPerThread;
        auto* smem_grad_sum = reinterpret_cast<Vec4TAcc<cache_t>*>(
            smem_.get_pointer().get()
        ) + warp_id * max_vecs * kThreadGroupSize;

        for (auto long_run_id = blockIdx_x; long_run_id < num_long_runs; long_run_id += gridDim_x) {
            // The first thread block in the really long run has run_id in long_run_ids
            // and the rest have the negative of its offset (see find_long_segments kernel).
            int32_t cta_rank_on_current_run = 0;
            int32_t current_run_id = long_run_ids_[long_run_id];
            if (current_run_id < 0) {
                cta_rank_on_current_run = -long_run_ids_[long_run_id];
                current_run_id = long_run_ids_[long_run_id - cta_rank_on_current_run];
            }
            const int32_t run_length =
                sorted_linear_indices_cumulative_run_lengths_[current_run_id + 1] -
                sorted_linear_indices_cumulative_run_lengths_[current_run_id];
            // This computation must agree with how we compute num_ctas_for_run in
            // find_long_segments kernel!
            const int32_t num_ctas_on_current_run =
                use_deterministic_algorithms_ ? 1 : div_round_up(run_length, max_segment_length_per_cta_);

            const int64_t linear_index = sorted_linear_indices_run_[current_run_id];
            const int32_t segment_start =
                sorted_linear_indices_cumulative_run_lengths_[current_run_id] +
                cta_rank_on_current_run * max_segment_length_per_cta_;
            const int32_t segment_end = std::min(
                use_deterministic_algorithms_ ? INT_MAX : segment_start + max_segment_length_per_cta_,
                sorted_linear_indices_cumulative_run_lengths_[current_run_id + 1]);
            const int32_t SL = segment_end - segment_start;

            const auto info_0 = sorted_infos_[segment_start];
            int32_t t_0 = info_0 % T;

            int64_t hash_size = hash_size_cumsum_[t_0];
            int64_t idx = linear_index - hash_size;

            const int32_t SL_per_warp = div_round_up(SL, static_cast<int32_t>(blockDim_y));
            const int32_t sl_start = SL_per_warp * warp_id;
            const int32_t sl_end = std::min(static_cast<int32_t>(SL_per_warp * (warp_id + 1)), SL);

            // Accumulate gradients (compute grad_sum)
            Vec4TAcc<cache_t> grad_sum[kFixedMaxVecsPerThread];
            constexpr int32_t kGroupVecWidth = kThreadGroupSize * VEC_WIDTH;
            const int32_t num_vecs = (D_ + kGroupVecWidth - 1) / kGroupVecWidth;

            compute_grad_sum_unweighted_nobag<
                grad_t,
                cache_t,
                kFixedMaxVecsPerThread,
                kThreadGroupSize,
                VEC_WIDTH,
                kUseVecBlocking>(
                    item,
                    grad_sum,
                    smem_grad_sum,
                    grad_output_,
                    D_,
                    T,
                    sorted_infos_,
                    segment_start,
                    sl_start,
                    sl_end,
                    num_vecs
            );

            // Do shared memory reduction only if we used multiple warps.
            if (SL > SL_per_warp) {
                item.barrier(sycl::access::fence_space::local_space);

                if (blockDim_y >= 32) {
                    if (warp_id < 16) {
                        for (int32_t vec = 0; vec < max_vecs && (vec * kThreadGroupSize + lane_id) * VEC_WIDTH < D_; ++vec) {
                            const int32_t d_vec = (vec * kThreadGroupSize + lane_id);
                            smem_grad_sum[d_vec] = vec4_acc(
                                smem_grad_sum[d_vec],
                                smem_grad_sum[d_vec + 16 * max_vecs * kThreadGroupSize]);
                        }
                    }
                    item.barrier(sycl::access::fence_space::local_space);
                }

                if (blockDim_y >= 16) {
                    if (warp_id < 8) {
                        for (int32_t vec = 0; vec < max_vecs && (vec * kThreadGroupSize + lane_id) * VEC_WIDTH < D_; ++vec) {
                            const int32_t d_vec = (vec * kThreadGroupSize + lane_id);
                            smem_grad_sum[d_vec] = vec4_acc(
                                smem_grad_sum[d_vec],
                                smem_grad_sum[d_vec + 8 * max_vecs * kThreadGroupSize]);
                        }
                    }
                    item.barrier(sycl::access::fence_space::local_space);
                }

                if (blockDim_y >= 8) {
                    if (warp_id < 4) {
                        for (int32_t vec = 0; vec < max_vecs && (vec * kThreadGroupSize + lane_id) * VEC_WIDTH < D_; ++vec) {
                            const int32_t d_vec = (vec * kThreadGroupSize + lane_id);
                            smem_grad_sum[d_vec] = vec4_acc(
                                smem_grad_sum[d_vec],
                                smem_grad_sum[d_vec + 4 * max_vecs * kThreadGroupSize]);
                        }
                    }
                    item.barrier(sycl::access::fence_space::local_space);
                }

                if (blockDim_y >= 4) {
                    if (warp_id < 2) {
                        for (int32_t vec = 0; vec < max_vecs && (vec * kThreadGroupSize + lane_id) * VEC_WIDTH < D_; ++vec) {
                            const int32_t d_vec = (vec * kThreadGroupSize + lane_id);
                            smem_grad_sum[d_vec] = vec4_acc(
                                smem_grad_sum[d_vec],
                                smem_grad_sum[d_vec + 2 * max_vecs * kThreadGroupSize]);
                        }
                    }
                    item.barrier(sycl::access::fence_space::local_space);
                }

                if (warp_id == 0) {
                    if constexpr (kUseVecBlocking) {
                        for (int32_t vec = 0;
                            vec < max_vecs &&
                            (kThreadGroupSize * vec + threadIdx_x) * VEC_WIDTH < D_;
                            ++vec) {
                            const int32_t d_vec = vec * kThreadGroupSize + threadIdx_x;
                            [[maybe_unused]] const int32_t d = d_vec * VEC_WIDTH;
                            smem_grad_sum[d_vec] = vec4_acc(
                                smem_grad_sum[d_vec],
                                smem_grad_sum[d_vec + max_vecs * kThreadGroupSize]);
                        }
                    } else {
                        #pragma unroll kFixedMaxVecsPerThread
                        for (int32_t vec = 0;
                            vec < kFixedMaxVecsPerThread
                                && (kThreadGroupSize * vec + threadIdx_x) * VEC_WIDTH < D_;
                            ++vec) {
                            const int32_t d_vec = vec * kThreadGroupSize + threadIdx_x;
                            [[maybe_unused]] const int32_t d = d_vec * VEC_WIDTH;
                            grad_sum[vec] = vec4_acc(
                                smem_grad_sum[d_vec],
                                smem_grad_sum[d_vec + max_vecs * kThreadGroupSize]);
                        }
                    }
                }
            }

            if (warp_id != 0) {
                continue;
            }

            if (num_ctas_on_current_run > 1) {
                int really_long_run_id = long_run_id_to_really_long_run_ids_[long_run_id];
                Vec4TAcc<cache_t> *temp_grad_accum_ptr =
                    reinterpret_cast<Vec4TAcc<cache_t>*>(&temp_grad_accum_[really_long_run_id][0]);

                if constexpr (kUseVecBlocking) {
                    for (int32_t vec = 0;
                        vec < max_vecs &&
                        (kThreadGroupSize * vec + threadIdx_x) * VEC_WIDTH < D_;
                        ++vec) {
                        const int32_t d_vec = vec * kThreadGroupSize + threadIdx_x;
                        [[maybe_unused]] const int32_t d = d_vec * VEC_WIDTH;
                        xpuAtomicAdd(&temp_grad_accum_ptr[d_vec].acc.x(), smem_grad_sum[d_vec].acc.x());
                        xpuAtomicAdd(&temp_grad_accum_ptr[d_vec].acc.y(), smem_grad_sum[d_vec].acc.y());
                        xpuAtomicAdd(&temp_grad_accum_ptr[d_vec].acc.z(), smem_grad_sum[d_vec].acc.z());
                        xpuAtomicAdd(&temp_grad_accum_ptr[d_vec].acc.w(), smem_grad_sum[d_vec].acc.w());
                    }
                } else {
                    #pragma unroll kFixedMaxVecsPerThread
                    for (int32_t vec = 0;
                        vec < kFixedMaxVecsPerThread
                            && (kThreadGroupSize * vec + threadIdx_x) * VEC_WIDTH < D_;
                        ++vec) {
                        const int32_t d_vec = vec * kThreadGroupSize + threadIdx_x;
                        [[maybe_unused]] const int32_t d = d_vec * VEC_WIDTH;
                        xpuAtomicAdd(&temp_grad_accum_ptr[d_vec].acc.x(), grad_sum[vec].acc.x());
                        xpuAtomicAdd(&temp_grad_accum_ptr[d_vec].acc.y(), grad_sum[vec].acc.y());
                        xpuAtomicAdd(&temp_grad_accum_ptr[d_vec].acc.z(), grad_sum[vec].acc.z());
                        xpuAtomicAdd(&temp_grad_accum_ptr[d_vec].acc.w(), grad_sum[vec].acc.w());
                    }
                }

                int counter = 0;
                if (threadIdx_x == 0) {
                    sycl::atomic_fence(sycl::memory_order::acq_rel, sycl::memory_scope::device);
                    counter = xpuAtomicAdd(&grad_accum_counter_[really_long_run_id], -1);
                }
                counter = sycl::group_broadcast(sg, counter, 0);
                // Only the thread block that accumulated last does the weight update.
                if (counter > 1) {
                    continue;
                }
                assert(counter == 1 && "Invalid grad_accum_counter. Race condition?");

                if constexpr (kUseVecBlocking) {
                    for (int32_t vec = 0;
                        vec < max_vecs &&
                        (kThreadGroupSize * vec + threadIdx_x) * VEC_WIDTH < D_;
                        ++vec) {
                        const int32_t d_vec = vec * kThreadGroupSize + threadIdx_x;
                        [[maybe_unused]] const int32_t d = d_vec * VEC_WIDTH;
                        smem_grad_sum[d_vec] = temp_grad_accum_ptr[d_vec];
                    }
                } else {
                    #pragma unroll kFixedMaxVecsPerThread
                    for (int32_t vec = 0;
                        vec < kFixedMaxVecsPerThread
                            && (kThreadGroupSize * vec + threadIdx_x) * VEC_WIDTH < D_;
                        ++vec) {
                        const int32_t d_vec = vec * kThreadGroupSize + threadIdx_x;
                        [[maybe_unused]] const int32_t d = d_vec * VEC_WIDTH;
                        grad_sum[vec] = temp_grad_accum_ptr[d_vec];
                    }
                }
            }

{%- if dense %}

            // Write deduplicated gradient to grad_dev_weights
            const int64_t weights_offset = weights_offsets_[t_0];
            store_grad_sum<
                emb_t,
                cache_t,
                kFixedMaxVecsPerThread,
                kThreadGroupSize,
                VEC_WIDTH,
                kUseVecBlocking>(
                    item,
                    grad_dev_weights_,
                    grad_sum,
                    kUseVecBlocking ? smem_grad_sum : nullptr,
                    D_,
                    weights_offset,
                    idx,
                    max_vecs
            );

{%- else %}

            // Apply rowwise adagrad optimizer update
            split_rowwise_adagrad_table_update_kernel<
                emb_t,
                cache_t,
                kFixedMaxVecsPerThread,
                kThreadGroupSize,
                VEC_WIDTH,
                kUseVecBlocking>(
                    dev_weights_,
                    uvm_weights_,
                    lxu_cache_weights_,
                    weights_placements_,
                    weights_offsets_,
                    sorted_lxu_cache_locations_,
                    grad_sum,
                    kUseVecBlocking ? smem_grad_sum : nullptr,
                    kIsInt8 ? smem_grad_sum : nullptr,
                    stochastic_rounding_,
                    stochastic_rounding_philox_args_,
                    current_run_id,
                    segment_start,
                    D_,
                    t_0,
                    idx,
                    1, // global_weight_decay
                    max_vecs,
                    momentum1_dev_,
                    momentum1_uvm_,
                    momentum1_placements_,
                    momentum1_offsets_,
                    item,
                    learning_rate_,
                    eps_,
                    weight_decay_,
                    weight_decay_mode_,
                    max_norm_
            );

{%- endif %}
        } // for each run
        }

    private:
        const at::PackedTensorAccessor64<grad_t, 2, RestrictPtrTraits> grad_output_;
        mutable at::PackedTensorAccessor64<emb_t, 1, RestrictPtrTraits> dev_weights_;
{%- if not dense %}
        mutable at::PackedTensorAccessor64<emb_t, 1, RestrictPtrTraits> uvm_weights_;
        mutable at::PackedTensorAccessor64<cache_t, 2, RestrictPtrTraits> lxu_cache_weights_;
        const at::PackedTensorAccessor32<int32_t, 1, RestrictPtrTraits> weights_placements_;
{%- endif %}
        const at::PackedTensorAccessor32<int64_t, 1, RestrictPtrTraits> weights_offsets_;
        int64_t D_;
        const at::PackedTensorAccessor32<int64_t, 1, RestrictPtrTraits> hash_size_cumsum_;
        const at::PackedTensorAccessor32<index_t, 1, RestrictPtrTraits> sorted_linear_indices_run_;
        const at::PackedTensorAccessor32<int32_t, 1, RestrictPtrTraits> sorted_linear_indices_cumulative_run_lengths_;
        const at::PackedTensorAccessor32<int32_t, 1, RestrictPtrTraits> long_run_ids_;
        const at::PackedTensorAccessor32<int32_t, 1, RestrictPtrTraits> num_long_run_ids_;
        const at::PackedTensorAccessor32<int64_t, 1, RestrictPtrTraits> sorted_infos_;
{%- if dense %}
        mutable at::PackedTensorAccessor64<emb_t, 1, RestrictPtrTraits> grad_dev_weights_;
{%- else %}
        const at::PackedTensorAccessor32<int32_t, 1, RestrictPtrTraits> sorted_lxu_cache_locations_;
        const bool use_uniq_cache_locations_;
        const at::PackedTensorAccessor32<int32_t, 1, RestrictPtrTraits> table_unique_indices_offsets_;
        bool stochastic_rounding_;
        PhiloxXpuState stochastic_rounding_philox_args_;
{%- endif %}
        const at::PackedTensorAccessor32<int32_t, 1, RestrictPtrTraits> long_run_id_to_really_long_run_ids_;
        mutable at::PackedTensorAccessor32<at::acc_type<cache_t, true>, 2, RestrictPtrTraits> temp_grad_accum_;
        mutable at::PackedTensorAccessor32<int32_t, 1, RestrictPtrTraits> grad_accum_counter_;
        int32_t max_segment_length_per_cta_;
        bool use_deterministic_algorithms_;
        int32_t max_vecs_per_thread_;
{%- if not dense %}
        mutable at::PackedTensorAccessor64<at::acc_type<cache_t, true>, 1, RestrictPtrTraits> momentum1_dev_;
        mutable at::PackedTensorAccessor64<at::acc_type<cache_t, true>, 1, RestrictPtrTraits> momentum1_uvm_;
        mutable at::PackedTensorAccessor32<int32_t, 1, RestrictPtrTraits> momentum1_placements_;
        mutable at::PackedTensorAccessor32<int64_t, 1, RestrictPtrTraits> momentum1_offsets_;
{%- endif %}
        sycl::local_accessor<cache_t, 1> smem_;
{%- if not dense %}
        float learning_rate_;
        float eps_;
        float weight_decay_;
        int64_t weight_decay_mode_;
        float max_norm_;
{%- endif %}
    };

} // namespace fbgemm_xpu
