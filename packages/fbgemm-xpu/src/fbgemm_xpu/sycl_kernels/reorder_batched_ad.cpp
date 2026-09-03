/*
 * Copyright (c) Meta Platforms, Inc. and affiliates. All rights reserved.
 * Copyright (c) 2026 Intel Corporation. All Rights Reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */

/*
 * SYCL/XPU Implementation of the reorder_batched_ad operators
 *
 * These operators reorder batched AD (advertisement) data from a batch-major
 * ragged layout to a table-major layout, which is the layout the downstream
 * embedding lookups expect.
 *
 * Input format (jagged/CSR-like, batch-major):
 *   cat_ad_lengths:  [B x T x #num_ads_b]  - length of each ad segment
 *   cat_ad_indices:  [seg0 data][seg1 data]...  - concatenated indices
 *   batch_offsets:   [B+1]  - first ad of each batch
 *
 * Output (table-major):
 *   reordered_cat_ad_lengths[t][b][a] = cat_ad_lengths[b][t][a]
 *   reordered_cat_ad_indices holds the indices of segment (t, b, a)
 *
 * In broadcast mode a single ad per (batch, table) is stored on the input side
 * and replicated across all ads of that batch on the output side.
 */

////////////////////////////////////////////////////////////////////////////////
// SYCL PORT MAPPING TO FBGEMM CUDA SOURCE - REORDER BATCHED AD OPERATORS
////////////////////////////////////////////////////////////////////////////////
//
// This file contains SYCL implementations of the FBGEMM batched AD reordering
// kernels and host functions.
//
// ORIGINAL CUDA SOURCE:
//   File: fbgemm_gpu/src/sparse_ops/sparse_reorder_batched_ad.cu
//
// KERNEL IMPLEMENTATIONS:
//   ReorderBatchedAdLengthsKernel::operator()
//     → reorder_batched_ad_lengths_kernel (CUDA)
//
//   NarrowBroadcastIndicesKernel::operator()
//     → narrow_broadcast_indices_kernel (CUDA)
//
//   NarrowBatchedBroadcastIndicesKernel::operator()
//     → narrow_batched_broadcast_indices_kernel (CUDA)
//
//   ReorderBatchedAdIndicesKernel::operator()
//     → reorder_batched_ad_indices_kernel (CUDA)
//
//   ReorderBatchedAdIndicesVecKernel::operator()
//     → reorder_batched_ad_indices_kernel_vec (CUDA)
//
// HOST FUNCTIONS:
//   reorder_batched_ad_lengths_xpu
//     → reorder_batched_ad_lengths_gpu (CUDA)
//
//   reorder_batched_ad_indices_xpu
//     → reorder_batched_ad_indices_gpu (CUDA)
//
////////////////////////////////////////////////////////////////////////////////

#include "reorder_batched_ad.h"
#include "fbgemm_utils/dispatch_macros.h"

#include <algorithm>
#include <cstdint>
#include <type_traits>

#include <ATen/xpu/XPUContext.h>
#include <c10/xpu/XPUFunctions.h>

namespace fbgemm_xpu {

// ============================================================================
// SYCL Kernel Functors - Operator Implementations
// ============================================================================

/**
 * @brief ReorderBatchedAdLengthsKernel operator implementation
 */
template <typename Dtype>
void ReorderBatchedAdLengthsKernel<Dtype>::operator()(
        const sycl::nd_item<2>& item) const {
    const int32_t B = batch_offsets_.size(0) - 1;

    const int32_t num_ads_in_batch = batch_offsets_[B];
    const int64_t BT = static_cast<int64_t>(B) * T_;

    const size_t b_t_init =
            item.get_group(0) * item.get_local_range(0) + item.get_local_id(0);

    const size_t stride = item.get_group_range(0) * item.get_local_range(0);

    for (size_t b_t = b_t_init; b_t < static_cast<size_t>(BT); b_t += stride) {
        const int32_t b = static_cast<int32_t>(b_t % B);
        const int32_t t = static_cast<int32_t>(b_t / B);

        const int32_t num_ads_b = batch_offsets_[b + 1] - batch_offsets_[b];
        const int32_t input_segment_start = broadcast_lengths_
                ? T_ * b + t
                : T_ * batch_offsets_[b] + t * num_ads_b;
        const int32_t output_segment_start = t * num_ads_in_batch + batch_offsets_[b];

        for (auto i = item.get_local_id(1); i < num_ads_b;
                  i += item.get_local_range(1)) {
            reordered_cat_ad_lengths_[output_segment_start + i] = broadcast_lengths_
                    ? cat_ad_lengths_[input_segment_start]
                    : cat_ad_lengths_[input_segment_start + i];
        }
    }
}

/**
 * @brief NarrowBroadcastIndicesKernel operator implementation
 */
template <typename Dtype, typename index_t>
void NarrowBroadcastIndicesKernel<Dtype, index_t>::operator()(
        const sycl::nd_item<1>& item) const {
    const auto lane_id = item.get_local_id(0) % kThreadGroupSize;
    const auto warp_id =
            (item.get_group(0) * item.get_local_range(0) + item.get_local_id(0)) /
            kThreadGroupSize;
    // The launch geometry rounds up to whole work-groups, so surplus warps are
    // started. Bail out before indexing cat_ad_offsets_, otherwise those warps
    // read out of bounds.
    if (warp_id >= reordered_cat_ad_batches_) {
        return;
    }
    const auto table_idx = warp_id / num_ads_in_batch_;
    const auto ads_idx = warp_id % num_ads_in_batch_;
    const auto start_offset = cat_ad_offsets_[table_idx];
    const auto end_offset = cat_ad_offsets_[table_idx + 1];
    const auto num_ads = end_offset - start_offset;
    for (auto i = lane_id; i < num_ads; i += kThreadGroupSize) {
        reordered_cat_ad_indices_
                [start_offset * num_ads_in_batch_ + ads_idx * num_ads + i] =
                        cat_ad_indices_[start_offset + i];
    }
}

/**
 * @brief NarrowBatchedBroadcastIndicesKernel operator implementation
 */
template <typename Dtype, typename index_t>
void NarrowBatchedBroadcastIndicesKernel<Dtype, index_t>::operator()(
        const sycl::nd_item<1>& item) const {
    const auto B = batch_offsets_.size(0) - 1;
    const auto num_ads_in_batch = static_cast<uint32_t>(batch_offsets_[B]);
    // calculate table_id and batch_id for this warp
    const auto warp_id =
            (item.get_group(0) * item.get_local_range(0) + item.get_local_id(0)) /
            static_cast<uint32_t>(kThreadGroupSize);
    const auto table_id = warp_id / num_ads_in_batch;
    const auto warp_id_in_table = warp_id % num_ads_in_batch;
    // warps in a table equally splited for each B
    const auto num_warp_in_batch = num_ads_in_batch / B;
    const auto batch_id = warp_id_in_table / num_warp_in_batch;
    if (table_id >= T_ || batch_id >= B) {
        return;
    }

    // all table_id and batch_id for this warp is the same
    const auto num_ads_b = batch_offsets_[batch_id + 1] - batch_offsets_[batch_id];
    const auto output_segment_offset_start =
            table_id * num_ads_in_batch + batch_offsets_[batch_id];
    const auto output_segment_start =
            reordered_cat_ad_offsets_[output_segment_offset_start];
    const auto input_segment_offset_start = T_ * batch_id + table_id;
    const auto input_segment_offset_end = input_segment_offset_start + 1;
    const auto input_segment_start = cat_ad_offsets_[input_segment_offset_start];
    const auto input_segment_end = cat_ad_offsets_[input_segment_offset_end];
    const auto num_elements = input_segment_end - input_segment_start;

    const auto warp_id_in_batch = warp_id_in_table % num_warp_in_batch;
    const auto lane_id_in_warp = item.get_local_id(0) % kThreadGroupSize;
    for (auto i = warp_id_in_batch; i < num_ads_b; i += num_warp_in_batch) {
        for (auto j = lane_id_in_warp; j < num_elements; j += kThreadGroupSize) {
            reordered_cat_ad_indices_[output_segment_start + i * num_elements + j] =
                    cat_ad_indices_[input_segment_start + j];
        }
    }
}

/**
 * @brief ReorderBatchedAdIndicesKernel operator implementation
 */
template <typename Dtype, typename index_t>
void ReorderBatchedAdIndicesKernel<Dtype, index_t>::operator()(
        const sycl::nd_item<2>& item) const {
    const int32_t B = batch_offsets_.size(0) - 1;
    const int32_t num_ads_in_batch = batch_offsets_[B];
    const int64_t BT = static_cast<int64_t>(B) * T_;
    const size_t b_t_init =
            item.get_group(0) * item.get_local_range(0) + item.get_local_id(0);
    const size_t stride = item.get_group_range(0) * item.get_local_range(0);
    for (size_t b_t = b_t_init; b_t < static_cast<size_t>(BT); b_t += stride) {
        const int32_t b = static_cast<int32_t>(b_t % B);
        const int32_t t = static_cast<int32_t>(b_t / B);

        const auto num_ads_b = batch_offsets_[b + 1] - batch_offsets_[b];
        const auto output_segment_offset_start =
                t * num_ads_in_batch + batch_offsets_[b];
        const auto output_segment_start =
                reordered_cat_ad_offsets_[output_segment_offset_start];
        const int32_t input_segment_offset_start =
                broadcast_indices_ ? T_ * b + t : T_ * batch_offsets_[b] + t * num_ads_b;
        const int32_t input_segment_offset_end = broadcast_indices_
                ? input_segment_offset_start + 1
                : input_segment_offset_start + num_ads_b;
        const auto input_segment_start = cat_ad_offsets_[input_segment_offset_start];
        const auto input_segment_end = cat_ad_offsets_[input_segment_offset_end];
        const auto num_elements = input_segment_end - input_segment_start;

        if (broadcast_indices_) {
            for (auto i = item.get_local_id(1); i < num_ads_b * num_elements;
                      i += item.get_local_range(1)) {
                reordered_cat_ad_indices_[output_segment_start + i] =
                        cat_ad_indices_[input_segment_start + i % num_elements];
            }
        } else {
            // Idea: we want to copy the entire segment of size sum_a(length_{b, t, a})
            // from starting point (given by cat_ad_offsets[b, t])
            // to end point (given by reordered_cat_ad_indices[t][b])
            for (auto i = item.get_local_id(1);
                      i < input_segment_end - input_segment_start;
                      i += item.get_local_range(1)) {
                reordered_cat_ad_indices_[output_segment_start + i] =
                        cat_ad_indices_[input_segment_start + i];
            }
        }
    }
}

/**
 * @brief ReorderBatchedAdIndicesVecKernel operator implementation
 *
 * Structural mirror of the CUDA reorder_batched_ad_indices_kernel_vec: same
 * branch layout (<=64 or non-4/8-byte dtype -> scalar, <=128 -> vec2,
 * >128 -> vec4), same alignment gate with a duplicated scalar fallback, and
 * the same tail handling.
 *
 * Two mechanical differences follow from SYCL rather than from the algorithm:
 *   - CUDA's threadIdx.x is the fastest-varying axis, SYCL's is the *last*
 *     nd_item dimension, so blockDim.y/threadIdx.y map to dimension 0 and
 *     blockDim.x/threadIdx.x to dimension 1.
 *   - sycl::long4 is naturally aligned to 32 bytes whereas CUDA's long4 is
 *     declared __align__(16), so the vec4 gate is stricter here for 8-byte
 *     dtypes. That only ever diverts a segment to the scalar fallback.
 */
template <typename Dtype, typename index_t>
void ReorderBatchedAdIndicesVecKernel<Dtype, index_t>::operator()(
        const sycl::nd_item<2>& item) const {
    using vec2_t =
            std::conditional_t<sizeof(Dtype) == 8, sycl::long2, sycl::float2>;
    using vec4_t =
            std::conditional_t<sizeof(Dtype) == 8, sycl::long4, sycl::float4>;
    const int32_t B = batch_offsets_.size(0) - 1;
    const int32_t num_ads_in_batch = batch_offsets_[B];
    const int64_t BT = static_cast<int64_t>(B) * T_;
    // warp-per-segment with grid-stride loop. A capped grid (used to stay
    // inside DPC++'s int32 work-item id limit) still covers all B*T segments.
    const size_t b_t_init =
            item.get_group(0) * item.get_local_range(0) + item.get_local_id(0);
    const size_t stride = item.get_group_range(0) * item.get_local_range(0);
    // CUDA threadIdx.x / blockDim.x.
    const size_t tid = item.get_local_id(1);
    const size_t nthreads = item.get_local_range(1);

    for (size_t b_t = b_t_init; b_t < static_cast<size_t>(BT); b_t += stride) {
        const int32_t b = static_cast<int32_t>(b_t % B);
        const int32_t t = static_cast<int32_t>(b_t / B);

        const auto num_ads_b = batch_offsets_[b + 1] - batch_offsets_[b];
        const auto output_segment_offset_start =
                t * num_ads_in_batch + batch_offsets_[b];
        const auto output_segment_start =
                reordered_cat_ad_offsets_[output_segment_offset_start];
        const int32_t input_segment_offset_start =
                broadcast_indices_ ? T_ * b + t : T_ * batch_offsets_[b] + t * num_ads_b;
        const int32_t input_segment_offset_end = broadcast_indices_
                ? input_segment_offset_start + 1
                : input_segment_offset_start + num_ads_b;
        const auto input_segment_start = cat_ad_offsets_[input_segment_offset_start];
        const auto input_segment_end = cat_ad_offsets_[input_segment_offset_end];
        const auto num_elements = input_segment_end - input_segment_start;

        Dtype* dst_ptr = &reordered_cat_ad_indices_[output_segment_start];
        const Dtype* src_ptr = &cat_ad_indices_[input_segment_start];
        if (broadcast_indices_) {
            for (auto i = tid; i < static_cast<size_t>(num_ads_b * num_elements);
                      i += nthreads) {
                reordered_cat_ad_indices_[output_segment_start + i] =
                        cat_ad_indices_[input_segment_start + i % num_elements];
            }
        } else {
            // Idea: we want to copy the entire segment of size sum_a(length_{b, t,
            // a}) from starting point (given by cat_ad_offsets[b, t]) to end point
            // (given by reordered_cat_ad_indices[t][b])
            if (num_elements <= 64 || !(sizeof(Dtype) == 4 || sizeof(Dtype) == 8)) {
                for (auto i = tid;
                          i < static_cast<size_t>(input_segment_end - input_segment_start);
                          i += nthreads) {
                    // coalesced global memory access, can be optimized through ILP
                    // with the help of shared memory or vector load/store (if
                    // num_ads_b>=64)
                    reordered_cat_ad_indices_[output_segment_start + i] =
                            cat_ad_indices_[input_segment_start + i];
                }
            } else if (num_elements > 64 && num_elements <= 128) {
                // Check alignment for vec2_t
                bool vec2_t_aligned =
                        reinterpret_cast<uintptr_t>(dst_ptr) % alignof(vec2_t) == 0 &&
                        reinterpret_cast<uintptr_t>(src_ptr) % alignof(vec2_t) == 0;
                if (vec2_t_aligned) {
                    // Use vectorized loads if properly aligned
                    auto dst = reinterpret_cast<vec2_t*>(dst_ptr);
                    auto src = reinterpret_cast<const vec2_t*>(src_ptr);
                    for (auto i = tid; i < static_cast<size_t>(num_elements / 2);
                              i += nthreads) {
                        dst[i] = src[i];
                    }
                    // Upstream hardcodes lane 31 here; the launch geometry gives
                    // this dimension exactly kThreadGroupSize work-items, so the
                    // last one is used instead of assuming a literal 31.
                    if ((num_elements % 2) && tid == nthreads - 1) {
                        reordered_cat_ad_indices_[output_segment_start + num_elements - 1] =
                                cat_ad_indices_[input_segment_start + num_elements - 1];
                    }
                } else {
                    // Fall back to scalar loads if misaligned
                    for (auto i = tid; i < static_cast<size_t>(num_elements);
                              i += nthreads) {
                        reordered_cat_ad_indices_[output_segment_start + i] =
                                cat_ad_indices_[input_segment_start + i];
                    }
                }
            } else if (num_elements > 128) {
                // Check alignment for vec4_t
                bool vec4_t_aligned =
                        reinterpret_cast<uintptr_t>(dst_ptr) % alignof(vec4_t) == 0 &&
                        reinterpret_cast<uintptr_t>(src_ptr) % alignof(vec4_t) == 0;
                if (vec4_t_aligned) {
                    // Use vectorized loads if properly aligned
                    auto dst = reinterpret_cast<vec4_t*>(dst_ptr);
                    auto src = reinterpret_cast<const vec4_t*>(src_ptr);
                    for (auto i = tid; i < static_cast<size_t>(num_elements / 4);
                              i += nthreads) {
                        dst[i] = src[i];
                    }
                    size_t remainder = static_cast<size_t>(num_elements % 4);
                    if (remainder && tid < remainder) {
                        reordered_cat_ad_indices_
                                [output_segment_start + num_elements - tid - 1] =
                                        cat_ad_indices_
                                                [input_segment_start + num_elements - tid - 1];
                    }
                } else {
                    // Fall back to scalar loads if misaligned
                    for (auto i = tid; i < static_cast<size_t>(num_elements);
                              i += nthreads) {
                        reordered_cat_ad_indices_[output_segment_start + i] =
                                cat_ad_indices_[input_segment_start + i];
                    }
                }
            }
        }
    }
}

// ============================================================================
// Host Functions - XPU Implementation
// ============================================================================

////////////////////////////////////////////////////////////////////////////////
// reorder_batched_ad_lengths_xpu - Host Function
////////////////////////////////////////////////////////////////////////////////
//
// CUDA SOURCE MAPPING:
//   CUDA Function: reorder_batched_ad_lengths_gpu
//   CUDA File: fbgemm_gpu/src/sparse_ops/sparse_reorder_batched_ad.cu
//
// DESCRIPTION:
//   Allocates the reordered lengths tensor and launches
//   ReorderBatchedAdLengthsKernel with one warp per (batch, table) segment.
//
////////////////////////////////////////////////////////////////////////////////

/**
 * @brief XPU implementation of reorder_batched_ad_lengths
 */
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

    const int64_t grid_size_uncapped = (B * T + 32 - 1) / 32;
    TORCH_CHECK(
        grid_size_uncapped >= 0,
        "grid_size must be positive, got ",
        grid_size_uncapped,
        " where B =",
        B,
        " and T =",
        T);

    const uint32_t grid_size = xpu_cap_grid_dim_x(grid_size_uncapped, 32 * 32);

    sycl::queue& queue = c10::xpu::getCurrentXPUStream().queue();
        FBGEMM_DISPATCH_ALL_TYPES(
                cat_ad_lengths.scalar_type(),
                "reorder_batched_ad_lengths_kernel_xpu",
                [&] {
                    queue.submit([&](sycl::handler& cgh) {
                        cgh.parallel_for<ReorderBatchedAdLengthsKernel<scalar_t>>(
                                sycl::nd_range<2>(
                                        sycl::range<2>(32 * grid_size, 32),
                                        sycl::range<2>(32, 32)),
                                ReorderBatchedAdLengthsKernel<scalar_t>(
                                        cat_ad_lengths.packed_accessor32<
                                                scalar_t,
                                                1,
                                                RestrictPtrTraits>(),
                                        batch_offsets.packed_accessor32<
                                                int32_t,
                                                1,
                                                RestrictPtrTraits>(),
                                        reordered_cat_ad_lengths.packed_accessor32<
                                                scalar_t,
                                                1,
                                                RestrictPtrTraits>(),
                                        T,
                                        broadcast_lengths));
                    });
                });

  return reordered_cat_ad_lengths;
}

////////////////////////////////////////////////////////////////////////////////
// reorder_batched_ad_indices_xpu - Host Function
////////////////////////////////////////////////////////////////////////////////
//
// CUDA SOURCE MAPPING:
//   CUDA Function: reorder_batched_ad_indices_gpu
//   CUDA File: fbgemm_gpu/src/sparse_ops/sparse_reorder_batched_ad.cu
//
// DESCRIPTION:
//   Allocates the reordered indices tensor and selects the kernel the same way
//   the CUDA reference does: the narrow broadcast kernels for
//   `broadcast_indices && T <= 320 && B < 64` (specialised further on B == 1),
//   otherwise the general ReorderBatchedAdIndicesKernel.
//
////////////////////////////////////////////////////////////////////////////////

/**
 * @brief XPU implementation of reorder_batched_ad_indices
 */
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

    sycl::queue& queue = c10::xpu::getCurrentXPUStream().queue();
    if (broadcast_indices && T <= 320 && B < 64) {
        TORCH_CHECK(num_ads_in_batch * T == reordered_cat_ad_offsets.numel() - 1);
        if (B == 1) {
            // for B = 1 broadcast case
            constexpr auto kNumWarps = 16;
            const int work_group_size = kNumWarps * kThreadGroupSize;
            const int global_dim =
                    xpu_calc_xblock_count(
                            reordered_cat_ad_offsets.numel() - 1, kNumWarps) *
                    work_group_size;
            FBGEMM_DISPATCH_ALL_TYPES(
                    cat_ad_indices.scalar_type(),
                    "narrow_broadcast_indices_kernel_1",
                    [&] {
                        AT_DISPATCH_INDEX_TYPES(
                                cat_ad_offsets.scalar_type(),
                                "narrow_broadcast_indices_kernel_2",
                                [&] {
                                    queue.submit([&](sycl::handler& cgh) {
                                        cgh.parallel_for<NarrowBroadcastIndicesKernel<
                                                scalar_t,
                                                index_t>>(
                                                sycl::nd_range<1>(
                                                        sycl::range<1>(global_dim),
                                                        sycl::range<1>(work_group_size)),
                                                NarrowBroadcastIndicesKernel<
                                                        scalar_t,
                                                        index_t>(
                                                        cat_ad_offsets.packed_accessor32<
                                                                index_t,
                                                                1,
                                                                RestrictPtrTraits>(),
                                                        cat_ad_indices.packed_accessor32<
                                                                scalar_t,
                                                                1,
                                                                RestrictPtrTraits>(),
                                                        reordered_cat_ad_indices
                                                                .packed_accessor32<
                                                                        scalar_t,
                                                                        1,
                                                                        RestrictPtrTraits>(),
                                                        num_ads_in_batch,
                                                        reordered_cat_ad_offsets.numel() - 1));
                                    });
                                });
                    });
            return reordered_cat_ad_indices;
        } else {
            // for B > 1 and B < 64 broadcast case
            constexpr auto kNumWarps = 16;
            const int work_group_size = kNumWarps * kThreadGroupSize;
            const int global_dim =
                    xpu_calc_xblock_count(T * num_ads_in_batch, kNumWarps) *
                    work_group_size;
            FBGEMM_DISPATCH_ALL_TYPES(
                    cat_ad_indices.scalar_type(),
                    "narrow_batched_broadcast_indices_kernel_1",
                    [&] {
                        AT_DISPATCH_INDEX_TYPES(
                                cat_ad_offsets.scalar_type(),
                                "narrow_batched_broadcast_indices_kernel_2",
                                [&] {
                                    queue.submit([&](sycl::handler& cgh) {
                                        cgh.parallel_for<
                                                NarrowBatchedBroadcastIndicesKernel<
                                                        scalar_t,
                                                        index_t>>(
                                                sycl::nd_range<1>(
                                                        sycl::range<1>(global_dim),
                                                        sycl::range<1>(work_group_size)),
                                                NarrowBatchedBroadcastIndicesKernel<
                                                        scalar_t,
                                                        index_t>(
                                                        cat_ad_offsets.packed_accessor32<
                                                                index_t,
                                                                1,
                                                                RestrictPtrTraits>(),
                                                        cat_ad_indices.packed_accessor32<
                                                                scalar_t,
                                                                1,
                                                                RestrictPtrTraits>(),
                                                        reordered_cat_ad_offsets
                                                                .packed_accessor32<
                                                                        index_t,
                                                                        1,
                                                                        RestrictPtrTraits>(),
                                                        reordered_cat_ad_indices
                                                                .packed_accessor32<
                                                                        scalar_t,
                                                                        1,
                                                                        RestrictPtrTraits>(),
                                                        batch_offsets.packed_accessor32<
                                                                int32_t,
                                                                1,
                                                                RestrictPtrTraits>(),
                                                        T));
                                    });
                                });
                    });
            return reordered_cat_ad_indices;
        }
    }
    FBGEMM_DISPATCH_ALL_TYPES(
            cat_ad_indices.scalar_type(),
            "reorder_batched_ad_indices_kernel_xpu_1",
            [&] {
                AT_DISPATCH_INDEX_TYPES(
                        cat_ad_offsets.scalar_type(),
                        "reorder_batched_ad_indices_kernel_xpu_2",
                        [&] {
                            constexpr auto kNumWarps = 32;
                            const int max_work_group_size = syclDeviceMaxWorkGroupSize();
                            auto max_warp_size = max_work_group_size / kNumWarps;
                            const int global_dim_y =
                                    max_warp_size < kThreadGroupSize ? max_warp_size : kThreadGroupSize;
                            const uint32_t num_groups = xpu_cap_grid_dim_x(
                                    xpu_calc_xblock_count(B * T, kNumWarps),
                                    static_cast<int64_t>(kNumWarps) * global_dim_y);
                            const uint32_t global_dim_x = num_groups * kNumWarps;
                            // Upstream launches the vectorized kernel for this
                            // path; it falls back to a scalar copy internally
                            // for short, misaligned or non-4/8-byte segments.
                            queue.submit([&](sycl::handler& cgh) {
                                cgh.parallel_for<ReorderBatchedAdIndicesVecKernel<
                                        scalar_t,
                                        index_t>>(
                                        sycl::nd_range<2>(
                                                sycl::range<2>(global_dim_x, global_dim_y),
                                                sycl::range<2>(kNumWarps, global_dim_y)),
                                        ReorderBatchedAdIndicesVecKernel<scalar_t, index_t>(
                                                cat_ad_offsets.packed_accessor32<
                                                        index_t,
                                                        1,
                                                        RestrictPtrTraits>(),
                                                cat_ad_indices.packed_accessor32<
                                                        scalar_t,
                                                        1,
                                                        RestrictPtrTraits>(),
                                                reordered_cat_ad_offsets.packed_accessor32<
                                                        index_t,
                                                        1,
                                                        RestrictPtrTraits>(),
                                                reordered_cat_ad_indices.packed_accessor32<
                                                        scalar_t,
                                                        1,
                                                        RestrictPtrTraits>(),
                                                batch_offsets.packed_accessor32<
                                                        int32_t,
                                                        1,
                                                        RestrictPtrTraits>(),
                                                T,
                                                broadcast_indices));
                            });
                        });
            });

  return reordered_cat_ad_indices;
}

// ============================================================================
// PyTorch Operator Registration
// ============================================================================

/**
 * Register XPU implementations with PyTorch dispatch system
 */
TORCH_LIBRARY_IMPL(fbgemm, XPU, m) {
  m.impl(
      "reorder_batched_ad_lengths",
      &fbgemm_xpu::reorder_batched_ad_lengths_xpu);
  m.impl(
      "reorder_batched_ad_indices",
      &fbgemm_xpu::reorder_batched_ad_indices_xpu);
}

} // namespace fbgemm_xpu
