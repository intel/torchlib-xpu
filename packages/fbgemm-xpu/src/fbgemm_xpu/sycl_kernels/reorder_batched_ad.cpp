/*
  * Copyright (c) Meta Platforms, Inc. and affiliates. All rights reserved.
  * Copyright (c) 2026 Intel Corporation. All Rights Reserved.
  * SPDX-License-Identifier: BSD-3-Clause
  */

#include "reorder_batched_ad.h"

namespace syclext = sycl::ext::oneapi;
namespace syclexp = sycl::ext::oneapi::experimental;

namespace fbgemm_xpu {

// ============================================================================
// SYCL Kernel Implementations
// ============================================================================

////////////////////////////////////////////////////////////////////////////////
// reorder_batched_ad_lengths_kernel_ - Device Kernel
////////////////////////////////////////////////////////////////////////////////
//
// CUDA SOURCE MAPPING:
//   CUDA Kernel: reorder_batched_ad_lengths_kernel
//   CUDA File: fbgemm_gpu/src/sparse_ops/sparse_reorder.cu
//
// DESCRIPTION:
//   Reorders AD lengths from ragged [B x T x #num_ads_b] to [T][B][#num_ads_b].
//   Each warp processes one (batch, table) pair and copies num_ads_b elements.
//   Supports broadcast mode where a single length is replicated for all ads.
//
////////////////////////////////////////////////////////////////////////////////
template <typename scalar_t>
SYCL_EXT_ONEAPI_FUNCTION_PROPERTY((syclexp::nd_range_kernel<2>))
void reorder_batched_ad_lengths_kernel_(
        const at::GenericPackedTensorAccessor<scalar_t, 1, RestrictPtrTraits, int32_t>
                cat_ad_lengths,
        const at::GenericPackedTensorAccessor<
                int32_t,
                1,
                RestrictPtrTraits,
                int32_t> batch_offsets,
        at::GenericPackedTensorAccessor<scalar_t, 1, RestrictPtrTraits, int32_t>
                reordered_cat_ad_lengths,
        const int32_t T,
        const bool broadcast_lengths) {
    const int32_t B = batch_offsets.size(0) - 1;

    const int32_t num_ads_in_batch = batch_offsets[B];
    // warp-per-segment.
    auto item = syclext::this_work_item::get_nd_item<2>();
    const auto b_t =
            item.get_group(0) * item.get_local_range(1) + item.get_local_id(1);
    const int32_t b = b_t % B;
    const int32_t t = b_t / B;
    if (t >= T) {
        return;
    }

    const int32_t num_ads_b = batch_offsets[b + 1] - batch_offsets[b];
    const int32_t input_segment_start =
            broadcast_lengths ? T * b + t : T * batch_offsets[b] + t * num_ads_b;
    const int32_t output_segment_start = t * num_ads_in_batch + batch_offsets[b];

    for (auto i = item.get_local_id(0); i < num_ads_b;
              i += item.get_local_range(0)) {
        reordered_cat_ad_lengths[output_segment_start + i] = broadcast_lengths
                ? cat_ad_lengths[input_segment_start]
                : cat_ad_lengths[input_segment_start + i];
    }
}

////////////////////////////////////////////////////////////////////////////////
// narrow_broadcast_indices_kernel_ - Device Kernel (B=1 optimization)
////////////////////////////////////////////////////////////////////////////////
//
// CUDA SOURCE MAPPING:
//   CUDA Kernel: narrow_broadcast_indices_kernel
//   CUDA File: fbgemm_gpu/src/sparse_ops/sparse_reorder.cu
//
// DESCRIPTION:
//   Optimized kernel for B=1 broadcast case. Each warp copies one ad segment
//   and broadcasts it to all output positions. Uses table-major iteration.
//
////////////////////////////////////////////////////////////////////////////////
template <typename scalar_t, typename index_t = int32_t>
SYCL_EXT_ONEAPI_FUNCTION_PROPERTY((syclexp::nd_range_kernel<1>))
void narrow_broadcast_indices_kernel_(
        const at::GenericPackedTensorAccessor<
                index_t,
                1,
                RestrictPtrTraits,
                int32_t> cat_ad_offsets,
        const at::GenericPackedTensorAccessor<scalar_t, 1, RestrictPtrTraits, int32_t>
                cat_ad_indices,
        at::GenericPackedTensorAccessor<scalar_t, 1, RestrictPtrTraits, int32_t>
                reordered_cat_ad_indices,
        const int num_ads_in_batch,
        const int reordered_cat_ad_batches,
        const int sub_group_size) {
    auto item = syclext::this_work_item::get_nd_item<1>();
    const auto lane_id = item.get_local_id(0) % sub_group_size;
    const auto warp_id =
            (item.get_group(0) * item.get_local_range(0) + item.get_local_id(0)) /
            sub_group_size;
    const auto table_idx = warp_id / num_ads_in_batch;
    const auto ads_idx = warp_id % num_ads_in_batch;
    const auto start_offset = cat_ad_offsets[table_idx];
    const auto end_offset = cat_ad_offsets[table_idx + 1];
    const auto num_ads = end_offset - start_offset;
    if (warp_id < reordered_cat_ad_batches) {
        for (auto i = lane_id; i < num_ads; i += sub_group_size) {
            reordered_cat_ad_indices
                    [start_offset * num_ads_in_batch + ads_idx * num_ads + i] =
                            cat_ad_indices[start_offset + i];
        }
    }
}

////////////////////////////////////////////////////////////////////////////////
// narrow_batched_broadcast_indices_kernel_ - Device Kernel (B>1 optimization)
////////////////////////////////////////////////////////////////////////////////
//
// CUDA SOURCE MAPPING:
//   CUDA Kernel: narrow_batched_broadcast_indices_kernel
//   CUDA File: fbgemm_gpu/src/sparse_ops/sparse_reorder.cu
//
// DESCRIPTION:
//   Optimized kernel for 1 < B < 64 broadcast case. Each warp handles one
//   (table, batch) pair and broadcasts indices from first batch to all ads
//   in that batch. More complex warp assignment than B=1 case.
//
////////////////////////////////////////////////////////////////////////////////
template <typename scalar_t, typename index_t = int32_t>
SYCL_EXT_ONEAPI_FUNCTION_PROPERTY((syclexp::nd_range_kernel<1>))
void narrow_batched_broadcast_indices_kernel_(
        const at::GenericPackedTensorAccessor<
                index_t,
                1,
                RestrictPtrTraits,
                int32_t> cat_ad_offsets,
        const at::GenericPackedTensorAccessor<scalar_t, 1, RestrictPtrTraits, int32_t>
                cat_ad_indices,
        const at::GenericPackedTensorAccessor<
                index_t,
                1,
                RestrictPtrTraits,
                int32_t> reordered_cat_ad_offsets,
        at::GenericPackedTensorAccessor<scalar_t, 1, RestrictPtrTraits, int32_t>
                reordered_cat_ad_indices,
        const at::GenericPackedTensorAccessor<
                int32_t,
                1,
                RestrictPtrTraits,
                int32_t> batch_offsets,
        const int32_t T,
        const int sub_group_size) {
    const auto B = batch_offsets.size(0) - 1;
    const auto num_ads_in_batch = static_cast<uint32_t>(batch_offsets[B]);
    // calculate table_id and batch_id for this warp
    auto item = syclext::this_work_item::get_nd_item<1>();
    const auto warp_id =
            (item.get_group(0) * item.get_local_range(0) + item.get_local_id(0)) /
            static_cast<uint32_t>(sub_group_size);
    const auto table_id = warp_id / num_ads_in_batch;
    const auto warp_id_in_table = warp_id % num_ads_in_batch;
    // warps in a table equally splited for each B
    const auto num_warp_in_batch = num_ads_in_batch / B;
    const auto batch_id = warp_id_in_table / num_warp_in_batch;
    if (table_id >= T || batch_id >= B) {
        return;
    }

    // all table_id and batch_id for this warp is the same
    const auto num_ads_b = batch_offsets[batch_id + 1] - batch_offsets[batch_id];
    const auto output_segment_offset_start =
            table_id * num_ads_in_batch + batch_offsets[batch_id];
    const auto output_segment_start =
            reordered_cat_ad_offsets[output_segment_offset_start];
    const auto input_segment_offset_start = T * batch_id + table_id;
    const auto input_segment_offset_end = input_segment_offset_start + 1;
    const auto input_segment_start = cat_ad_offsets[input_segment_offset_start];
    const auto input_segment_end = cat_ad_offsets[input_segment_offset_end];
    const auto num_elements = input_segment_end - input_segment_start;

    const auto warp_id_in_batch = warp_id_in_table % num_warp_in_batch;
    const auto lane_id_in_warp = item.get_local_id(0) % sub_group_size;
    for (auto i = warp_id_in_batch; i < num_ads_b; i += num_warp_in_batch) {
        for (auto j = lane_id_in_warp; j < num_elements; j += sub_group_size) {
            reordered_cat_ad_indices[output_segment_start + i * num_elements + j] =
                    cat_ad_indices[input_segment_start + j];
        }
    }
}

////////////////////////////////////////////////////////////////////////////////
// reorder_batched_ad_indices_kernel_ - Device Kernel (General case)
////////////////////////////////////////////////////////////////////////////////
//
// CUDA SOURCE MAPPING:
//   CUDA Kernel: reorder_batched_ad_indices_kernel
//   CUDA File: fbgemm_gpu/src/sparse_ops/sparse_reorder.cu
//
// DESCRIPTION:
//   General kernel for reordering indices from [B x T x #num_ads_b x L] to
//   [T][B][#num_ads_b][L]. Each warp processes one (batch, table) pair and
//   copies all indices for all ads in that segment. Handles both broadcast
//   and non-broadcast modes.
//
////////////////////////////////////////////////////////////////////////////////
template <typename scalar_t, typename index_t = int32_t>
SYCL_EXT_ONEAPI_FUNCTION_PROPERTY((syclexp::nd_range_kernel<2>))
void reorder_batched_ad_indices_kernel_(
        const at::GenericPackedTensorAccessor<
                index_t,
                1,
                RestrictPtrTraits,
                int32_t> cat_ad_offsets,
        const at::GenericPackedTensorAccessor<scalar_t, 1, RestrictPtrTraits, int32_t>
                cat_ad_indices,
        const at::GenericPackedTensorAccessor<
                index_t,
                1,
                RestrictPtrTraits,
                int32_t> reordered_cat_ad_offsets,
        at::GenericPackedTensorAccessor<scalar_t, 1, RestrictPtrTraits, int32_t>
                reordered_cat_ad_indices,
        const at::GenericPackedTensorAccessor<
                int32_t,
                1,
                RestrictPtrTraits,
                int32_t> batch_offsets,
        const int32_t T,
        const bool broadcast_indices) {
    const int32_t B = batch_offsets.size(0) - 1;
    const int32_t num_ads_in_batch = batch_offsets[B];
    // warp-per-segment.
    auto item = syclext::this_work_item::get_nd_item<2>();
    const auto b_t =
            item.get_group(0) * item.get_local_range(1) + item.get_local_id(1);
    const int32_t b = b_t % B;
    const int32_t t = b_t / B;
    if (t >= T) {
        return;
    }

    const auto num_ads_b = batch_offsets[b + 1] - batch_offsets[b];
    const auto output_segment_offset_start =
            t * num_ads_in_batch + batch_offsets[b];
    const auto output_segment_start =
            reordered_cat_ad_offsets[output_segment_offset_start];
    const int32_t input_segment_offset_start =
            broadcast_indices ? T * b + t : T * batch_offsets[b] + t * num_ads_b;
    const int32_t input_segment_offset_end = broadcast_indices
            ? input_segment_offset_start + 1
            : input_segment_offset_start + num_ads_b;
    const auto input_segment_start = cat_ad_offsets[input_segment_offset_start];
    const auto input_segment_end = cat_ad_offsets[input_segment_offset_end];
    const auto num_elements = input_segment_end - input_segment_start;

    if (broadcast_indices) {
        for (auto i = item.get_local_id(0); i < num_ads_b * num_elements;
                  i += item.get_local_range(0)) {
            reordered_cat_ad_indices[output_segment_start + i] =
                    cat_ad_indices[input_segment_start + i % num_elements];
        }
    } else {
        // Idea: we want to copy the entire segment of size sum_a(length_{b, t, a})
        // from starting point (given by cat_ad_offsets[b, t])
        // to end point (given by reordered_cat_ad_indices[t][b])
        for (auto i = item.get_local_id(0);
                  i < input_segment_end - input_segment_start;
                  i += item.get_local_range(0)) {
            reordered_cat_ad_indices[output_segment_start + i] =
                    cat_ad_indices[input_segment_start + i];
        }
    }
}

// ============================================================================
// Host Kernel Dispatcher Functions
// ============================================================================

void reorder_batched_ad_lengths_kernel_xpu(
        const at::Tensor& cat_ad_lengths,
        const at::Tensor& batch_offsets,
        at::Tensor& reordered_cat_ad_lengths,
        const int32_t T,
        const bool broadcast_lengths,
        const int32_t grid_size) {
    FBGEMM_DISPATCH_ALL_TYPES(
            cat_ad_lengths.scalar_type(),
            "reorder_batched_ad_lengths_kernel_xpu",
            [&] {
                sycl_kernel_submit<reorder_batched_ad_lengths_kernel_<scalar_t>>(
                        sycl::range<2>(32 * grid_size, 32),
                        sycl::range<2>(32, 32),
                        at::xpu::getCurrentSYCLQueue(),
                        0,
                        cat_ad_lengths
                                .packed_accessor32<scalar_t, 1, RestrictPtrTraits>(),
                        batch_offsets
                                .packed_accessor32<int32_t, 1, RestrictPtrTraits>(),
                        reordered_cat_ad_lengths
                                .packed_accessor32<scalar_t, 1, RestrictPtrTraits>(),
                        T,
                        broadcast_lengths);
            });
}

void reorder_batched_ad_indices_kernel_xpu(
        const at::Tensor& cat_ad_offsets,
        const at::Tensor& cat_ad_indices,
        const at::Tensor& reordered_cat_ad_offsets,
        const at::Tensor& batch_offsets,
        at::Tensor& reordered_cat_ad_indices,
        const int64_t num_ads_in_batch,
        const int64_t B,
        const int64_t T,
        const bool broadcast_indices) {
    const int sub_group_size = syclMaxSubGroupSize();
    if (broadcast_indices && T <= 320 && B < 64) {
        TORCH_CHECK(num_ads_in_batch * T == reordered_cat_ad_offsets.numel() - 1);
        if (B == 1) {
            // for B = 1 broadcast case
            constexpr auto kNumWarps = 16;
            const int work_group_size = kNumWarps * sub_group_size;
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
                                    sycl_kernel_submit<
                                            narrow_broadcast_indices_kernel_<scalar_t, index_t>>(
                                            sycl::range<1>(global_dim),
                                            sycl::range<1>(work_group_size),
                                            at::xpu::getCurrentSYCLQueue(),
                                            0,
                                            cat_ad_offsets.packed_accessor32<
                                                    index_t,
                                                    1,
                                                    RestrictPtrTraits>(),
                                            cat_ad_indices.packed_accessor32<
                                                    scalar_t,
                                                    1,
                                                    RestrictPtrTraits>(),
                                            reordered_cat_ad_indices.packed_accessor32<
                                                    scalar_t,
                                                    1,
                                                    RestrictPtrTraits>(),
                                            num_ads_in_batch,
                                            reordered_cat_ad_offsets.numel() - 1,
                                            sub_group_size);
                                });
                    });
            return;
        } else {
            // for B > 1 and B < 64 broadcast case
            constexpr auto kNumWarps = 16;
            const int work_group_size = kNumWarps * sub_group_size;
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
                                    sycl_kernel_submit<narrow_batched_broadcast_indices_kernel_<
                                            scalar_t,
                                            index_t>>(
                                            sycl::range<1>(global_dim),
                                            sycl::range<1>(work_group_size),
                                            at::xpu::getCurrentSYCLQueue(),
                                            0,
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
                                            sub_group_size);
                                });
                    });
            return;
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
                                    max_warp_size < sub_group_size ? max_warp_size : sub_group_size;
                            const int global_dim_x =
                                    xpu_calc_xblock_count(B * T, kNumWarps) * kNumWarps;
                            sycl_kernel_submit<
                                    reorder_batched_ad_indices_kernel_<scalar_t, index_t>>(
                                    sycl::range<2>(global_dim_x, global_dim_y),
                                    sycl::range<2>(kNumWarps, global_dim_y),
                                    at::xpu::getCurrentSYCLQueue(),
                                    0,
                                    cat_ad_offsets
                                            .packed_accessor32<index_t, 1, RestrictPtrTraits>(),
                                    cat_ad_indices
                                            .packed_accessor32<scalar_t, 1, RestrictPtrTraits>(),
                                    reordered_cat_ad_offsets
                                            .packed_accessor32<index_t, 1, RestrictPtrTraits>(),
                                    reordered_cat_ad_indices
                                            .packed_accessor32<scalar_t, 1, RestrictPtrTraits>(),
                                    batch_offsets
                                            .packed_accessor32<int32_t, 1, RestrictPtrTraits>(),
                                    T,
                                    broadcast_indices);
                        });
            });
}

} // namespace fbgemm_xpu
