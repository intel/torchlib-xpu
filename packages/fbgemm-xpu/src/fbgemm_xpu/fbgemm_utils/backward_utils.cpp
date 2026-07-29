/*
 * Copyright (c) Meta Platforms, Inc. and affiliates. All rights reserved.
 * Copyright (c) 2026 Intel Corporation. All Rights Reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */

#include <tuple>

#include "fbgemm_utils/backward_utils.h"

namespace fbgemm_xpu {

// oneDPL's default kernel naming can collide when this sort callsite is
// instantiated multiple times via dispatch templates. Give it an explicit,
// type-dependent kernel name to keep device symbols unique.
template <typename key_t, typename value_t>
class TransposeEmbeddingInputSortKernel;

void SplitEmbeddingBackwardFindLongSegments::operator()(
    const sycl::nd_item<1>& item) const {
    const auto threadIdx_x = item.get_local_id(0);
    const auto blockIdx_x = item.get_group(0);
    const auto blockDim_x = item.get_local_range(0);
    const auto gridDim_x = item.get_group_range(0);

    const int32_t num_runs = sorted_linear_indices_num_runs_[0];

    for (auto run_id = blockIdx_x * blockDim_x + threadIdx_x; run_id < num_runs;
         run_id += blockDim_x * gridDim_x) {
        if (sorted_linear_indices_run_lengths_[run_id] >=
            max_segment_length_per_warp_) {
            // A segment with length > max_segment_length_per_cta is handled by
            // more than 1 thread block.
            const int num_ctas_for_run =
                use_deterministic_algorithms_
                    ? 1
                    : div_round_up(sorted_linear_indices_run_lengths_[run_id],
                                   max_segment_length_per_cta_);

            const auto long_run_idx =
                xpuAtomicAdd(&num_long_run_ids_[0], num_ctas_for_run);
            // The first thread block in the really long run gets run_id in
            // long_run_ids and the rest get the negative of its offset.
            long_run_ids_[long_run_idx] = run_id;
            for (int i = 1; i < num_ctas_for_run; ++i) {
                long_run_ids_[long_run_idx + i] = -i;
            }
            if (num_ctas_for_run > 1) {
                const auto really_long_run_idx =
                    xpuAtomicAdd(&num_really_long_run_ids_[0], 1);
                grad_accum_counter_[really_long_run_idx] = num_ctas_for_run;
                for (int i = 0; i < num_ctas_for_run; ++i) {
                    long_run_id_to_really_long_run_ids_[long_run_idx + i] =
                        really_long_run_idx;
                }
            }
        }
    }
}

// Pass 1: Mark run starts
template <typename index_t>
void MarkRunStartsKernel<index_t>::operator()(
    const sycl::nd_item<1>& item) const {
    const auto tid = item.get_global_id(0);

    if (tid >= total_elements_)
        return;

    // Check if this is the start of a new run
    bool is_run_start =
        (tid == 0) || (sorted_input_[tid] != sorted_input_[tid - 1]);
    run_starts_[tid] = is_run_start ? 1 : 0;
}

// Pass 2: Compact runs using prefix sum
template <typename index_t>
void CompactRunsKernel<index_t>::operator()(
    const sycl::nd_item<1>& item) const {
    const auto tid = item.get_global_id(0);

    if (tid >= total_elements_) {
        return;
    }

    if (run_starts_[tid] == 1) {
        // Use the prefix-sum information in run_positions to locate the end of
        // this run. run_positions is a non-decreasing prefix sum over
        // run_starts, so each increment corresponds to a new run. For a run
        // starting at tid with index run_idx, the next run (if any) starts at
        // the first position > tid where run_positions[pos] > run_idx.
        const int32_t run_idx =
            run_positions_[tid];  // Position from prefix sum

        int64_t left = static_cast<int64_t>(tid) + 1;
        int64_t right = static_cast<int64_t>(total_elements_);
        int64_t run_end = right;

        while (left < right) {
            int64_t mid = left + ((right - left) >> 1);
            if (run_positions_[mid] > run_idx) {
                run_end = mid;
                right = mid;
            } else {
                left = mid + 1;
            }
        }

        int32_t run_length =
            static_cast<int32_t>(run_end - static_cast<int64_t>(tid));
        unique_output_[run_idx] = sorted_input_[tid];
        run_lengths_[run_idx] = run_length;
    }
}

template <typename index_t, typename info_acc_t, bool nobag, bool vbe>
void LinearizeIndexKernel<index_t, info_acc_t, nobag, vbe>::operator()(
    const sycl::nd_item<1>& item) const {
    const auto threadIdx_x = item.get_local_id(0);
    const auto blockIdx_x = item.get_group(0);
    const auto blockDim_x = item.get_local_range(0);
    const auto sg = item.get_sub_group();

    // Print from first work-item only to avoid spam
    const auto T = hash_size_cumsum_.size(0) - 1;

    auto b_t = blockIdx_x * blockDim_x + threadIdx_x;
    int32_t b;
    int32_t t;
    const auto total_B = offsets_.size(0) - 1;
    bool valid = b_t < total_B;
    // info must be uint32_t (using auto will assign int32_t to info)
    uint32_t info = 0;

    if (vbe && valid) {
        info = vbe_b_t_map_[b_t];
        reinterpret_cast<uint32_t*>(&t)[0] = info >> info_B_num_bits_;
        reinterpret_cast<uint32_t*>(&b)[0] = info & info_B_mask_;
    } else {
        fd_.DivMod(b_t, &t, &b);
    }

    const index_t hash_offset = valid ? hash_size_cumsum_[t] : -1;
    const auto indices_start = valid ? offsets_[b_t] : -1;
    const auto L = valid ? offsets_[b_t + 1] - indices_start : 0;
    const uint32_t lane_id = threadIdx_x % kThreadGroupSize;

    // Compile-time conditional
    if (nobag) {
        for (int32_t j = 0; j < kThreadGroupSize; ++j) {
            const auto indices_start_warp =
                sycl::select_from_group(sg, indices_start, j);
            const auto t_warp = sycl::select_from_group(sg, t, j);
            const auto L_warp = sycl::select_from_group(sg, L, j);
            const index_t hash_offset_warp =
                sycl::select_from_group(sg, hash_offset, j);

            for (auto i = lane_id; i < L_warp; i += kThreadGroupSize) {
                const index_t idx = indices_[indices_start_warp + i];
                const auto l_t = (indices_start_warp + i) * T + t_warp;
                infos_[indices_start_warp + i] = l_t;
                linear_indices_[indices_start_warp + i] =
                    hash_offset_warp + idx;
            }
        }
    } else {
        // Store t in upper (32 - kDefaultInfoBNumBits).
        // Store b in lower (kDefaultInfoBNumBits).
        if (!vbe && valid) {
            info = (reinterpret_cast<uint32_t*>(&t)[0] << info_B_num_bits_) |
                   reinterpret_cast<uint32_t*>(&b)[0];
        }
        for (int32_t j = 0; j < kThreadGroupSize; ++j) {
            const auto indices_start_warp =
                sycl::select_from_group(sg, indices_start, j);
            const auto info_warp = sycl::select_from_group(sg, info, j);
            const auto L_warp = sycl::select_from_group(sg, L, j);
            const index_t hash_offset_warp =
                sycl::select_from_group(sg, hash_offset, j);
            for (int32_t i = lane_id; i < L_warp; i += kThreadGroupSize) {
                const index_t idx = indices_[indices_start_warp + i];
                reinterpret_cast<uint32_t*>(
                    &infos_[0])[indices_start_warp + i] = info_warp;
                linear_indices_[indices_start_warp + i] =
                    hash_offset_warp + idx;
            }
        }
    }
}

std::tuple<Tensor /*linear_indices*/, Tensor /*linear_indices_sorted*/, \
           Tensor /*infos_sorted*/, Tensor /*sorted_linear_indices_run*/, \
           Tensor /*sorted_linear_indices_run_lengths*/, \
           Tensor /*sorted_linear_indices_num_runs*/, \
           Tensor /*sorted_linear_indices_cumulative_run_lengths*/>
transpose_embedding_input(
    Tensor hash_size_cumsum, int64_t total_hash_size_bits, Tensor indices,
    Tensor offsets, bool nobag, const std::optional<Tensor>& vbe_b_t_map,
    const int64_t info_B_num_bits, const int64_t info_B_mask,
    const int64_t total_unique_indices, const bool is_index_select,
    const std::optional<Tensor>& total_L_offsets,
    const int64_t fixed_L_per_warp, const int64_t num_warps_per_feature) {
    const bool vbe = vbe_b_t_map.has_value();

    TORCH_CHECK(nobag || !vbe || info_B_num_bits > 0);
    TORCH_CHECK(!vbe || info_B_mask > 0);
    TORCH_CHECK(!is_index_select ||
                (fixed_L_per_warp > 0 && num_warps_per_feature > 0));

    const auto T = hash_size_cumsum.size(0) - 1;
    const auto total_B =
        !is_index_select ? (offsets.size(0) - 1) : (num_warps_per_feature * T);

    TORCH_CHECK(!is_index_select || (total_L_offsets.has_value() &&
                                     total_L_offsets.value().numel() == T + 1));

    auto infos = at::empty_like(
        indices, indices.options().dtype(
                     (nobag || is_index_select) ? at::kLong : at::kInt));
    auto infos_sorted = at::empty_like(infos);
    auto linear_indices = at::empty_like(indices);
    auto linear_indices_sorted = at::empty_like(indices);

    Tensor sorted_linear_indices_run;
    Tensor sorted_linear_indices_run_lengths;
    Tensor sorted_linear_indices_num_runs;

    AT_DISPATCH_INDEX_TYPES(
        infos.scalar_type(), "transpose_embedding_input_1", [&] {
            AT_DISPATCH_INDEX_TYPES(
                indices.scalar_type(), "transpose_embedding_input_2", [&] {
                    if (!is_index_select) {
                        if (!nobag) {
                            TORCH_CHECK(false,
                                        "linearize_index_kernel kernel for bag "
                                        "operations not implemented yet in "
                                        "SYCL backend");
                        } else {
                            size_t local_range = kMaxThreads;
                            size_t global_range =
                                div_round_up(static_cast<size_t>(total_B),
                                             local_range) *
                                local_range;

                            sycl::queue& queue =
                                c10::xpu::getCurrentXPUStream().queue();
                            queue.submit([&](sycl::handler& cgh) {
                                cgh.parallel_for<LinearizeIndexKernel<
                                    index_t, int64_t, true, false>>(
                                    sycl::nd_range<1>(global_range,
                                                      local_range),
                                    LinearizeIndexKernel<index_t, int64_t, true,
                                                         false>(
                                        hash_size_cumsum.packed_accessor32<
                                            int64_t, 1, RestrictPtrTraits>(),
                                        indices.packed_accessor32<
                                            index_t, 1, RestrictPtrTraits>(),
                                        offsets.packed_accessor32<
                                            index_t, 1, RestrictPtrTraits>(),
                                        infos.packed_accessor32<
                                            int64_t, 1, RestrictPtrTraits>(),
                                        linear_indices.packed_accessor32<
                                            index_t, 1, RestrictPtrTraits>(),
                                        info_B_num_bits, info_B_mask,
                                        (1u << (kDefaultInfoNumBits -
                                                info_B_num_bits)) -
                                            1,
                                        (1u << info_B_num_bits) - 1, nullptr,
                                        FixedDivisor(total_B / T)));
                            });
                        }
                    } else {
                        TORCH_CHECK(
                            false,
                            "linearize_index_index_select_kernel kernel not "
                            "implemented yet in SYCL backend");
                    }
                    {
                        auto sort_result = at::sort(linear_indices, /*dim=*/0,
                                                    /*descending=*/false);
                        auto sorted_keys = std::get<0>(sort_result);
                        auto perm = std::get<1>(
                            sort_result);  // permutation indices (int64)

                        linear_indices_sorted.copy_(sorted_keys);
                        infos_sorted.copy_(infos.index_select(0, perm));
                    }
                    if (total_unique_indices != -1) {
                        TORCH_CHECK(total_unique_indices >= 0);
                        sorted_linear_indices_run = at::empty(
                            {total_unique_indices}, indices.options());
                        sorted_linear_indices_run_lengths =
                            at::zeros({total_unique_indices},
                                      indices.options().dtype(at::kInt));
                    } else {
                        sorted_linear_indices_run = at::empty_like(indices);
                        sorted_linear_indices_run_lengths = at::zeros_like(
                            indices, indices.options().dtype(at::kInt));
                    }
                    sorted_linear_indices_num_runs =
                        at::zeros({1}, indices.options().dtype(at::kInt));

                    {
                        // Run-length encoding using custom SYCL kernel
                        sycl::queue& queue =
                            c10::xpu::getCurrentXPUStream().queue();
                        const int64_t n = linear_indices_sorted.numel();

                        // Temporary buffer to mark run starts (1 for start, 0
                        // otherwise)
                        auto run_starts =
                            at::zeros({n}, indices.options().dtype(at::kInt));

                        // Pass 1: Mark run starts
                        size_t local_range = kMaxThreads;
                        size_t global_range =
                            div_round_up(static_cast<size_t>(n), local_range) *
                            local_range;

                        queue.submit([&](sycl::handler& cgh) {
                            cgh.parallel_for<MarkRunStartsKernel<index_t>>(
                                sycl::nd_range<1>(global_range, local_range),
                                MarkRunStartsKernel<index_t>(
                                    linear_indices_sorted.data_ptr<index_t>(),
                                    run_starts.data_ptr<int32_t>(), n));
                        });

                        // Compute prefix sum to get positions for each run
                        // start cumsum gives us [1, 1, 1, 2, 2, 3, ...] for
                        // positions [0, 1, 2, 3, 4, 5, ...] Subtract 1 to get
                        // 0-indexed positions: [0, 0, 0, 1, 1, 2, ...]
                        auto run_positions = at::cumsum(run_starts, /*dim=*/0,
                                                        /*dtype=*/at::kInt)
                                                 .sub_(1);

                        // Get total number of runs (last position + 1)
                        int32_t total_runs =
                            run_positions[-1].item<int32_t>() + 1;
                        sorted_linear_indices_num_runs.fill_(total_runs);

                        // Pass 2: Compact runs using the computed positions
                        queue.submit([&](sycl::handler& cgh) {
                            cgh.parallel_for<CompactRunsKernel<index_t>>(
                                sycl::nd_range<1>(global_range, local_range),
                                CompactRunsKernel<index_t>(
                                    linear_indices_sorted.data_ptr<index_t>(),
                                    run_starts.data_ptr<int32_t>(),
                                    run_positions.data_ptr<int32_t>(),
                                    sorted_linear_indices_run
                                        .data_ptr<index_t>(),
                                    sorted_linear_indices_run_lengths
                                        .data_ptr<int32_t>(),
                                    n));
                        });
                    }
                });
        });

    auto sorted_linear_indices_cumulative_run_lengths =
        asynchronous_complete_cumsum_xpu(sorted_linear_indices_run_lengths);

    return {linear_indices,
            linear_indices_sorted,
            infos_sorted,
            sorted_linear_indices_run,
            sorted_linear_indices_run_lengths,
            sorted_linear_indices_num_runs,
            sorted_linear_indices_cumulative_run_lengths};
}
}  // namespace fbgemm_xpu
