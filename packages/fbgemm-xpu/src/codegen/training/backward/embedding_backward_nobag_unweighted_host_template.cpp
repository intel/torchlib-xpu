/*
 * Copyright (c) Meta Platforms, Inc. and affiliates. All rights reserved.
 * Copyright (c) 2026 Intel Corporation. All Rights Reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */


{#
// @lint-ignore LINTIGNORE
// @lint-ignore-every CLANGFORMAT
// clang-format off
// Note: clang-format off doesn't work with this templated code,
// so we need to keep lint-ignore-every.
#}

{%- set optimizer = "Dense" if dense else "RowwiseAdagrad" %}
{%- set host_fn = "dense" if dense else "rowwise_adagrad" %}
{%- set find_segs_kernel = "Dense" if dense else "Split" %}

#include "../fbgemm_utils/dispatch_macros.h"

////////////////////////////////////////////////////////////////////////////////
// SYCL PORT MAPPING TO FBGEMM CUDA SOURCE - BACKWARD HOST DISPATCH
////////////////////////////////////////////////////////////////////////////////
//
// This file contains the SYCL port of the FBGEMM {{ "dense" if dense else "split" }} embedding
// backward host dispatch function for no-bag (sequence) unweighted lookups
// with {{ "no" if dense else optimizer }} optimizer.
//
// ORIGINAL CUDA SOURCE:
//   Template: fbgemm_gpu/codegen/training/backward/embedding_backward_split_template.cu
//   Generated Source: fbgemm_gpu/_skbuild/linux-x86_64-3.10/cmake-build/gen_embedding_backward_{{ "dense" if dense else "rowwise_adagrad" }}_split_unweighted_exact_cuda.cu
//
// HOST FUNCTION MAPPING:
//   split_embedding_nobag_backward_codegen_{{ host_fn }}_unweighted_exact_xpu
//     → split_embedding_nobag_backward_codegen_{{ host_fn }}_unweighted_exact_cuda (CUDA)
//
// KERNEL DISPATCH STRATEGY:
//   Step 1 - Segment classification (preprocessing):
//     SplitEmbeddingBackwardFindLongSegments{{ find_segs_kernel }}Kernel
//       Classifies each run (unique embedding index) as short or long based on
//       max_segment_length_per_warp and fills long_run_ids for the CTA kernel.
//
//   Step 2 - Long segments (SL >= max_segment_length_per_warp):
//     SplitEmbeddingNobagBackwardCodegen{{ optimizer }}UnweightedKernelCtaPerRow
//       One CTA processes all gradient contributions for one embedding row.
//       Source: gen_embedding_backward_{{ "dense" if dense else "rowwise_adagrad" }}_split_unweighted_nobag_kernels.h
//
//   Step 3 - Short segments (SL < max_segment_length_per_warp):
//     SplitEmbeddingNobagBackwardCodegen{{ optimizer }}UnweightedKernelWarpPerRow
//       One sub-group (warp) processes one embedding row.
//       Source: gen_embedding_backward_{{ "dense" if dense else "rowwise_adagrad" }}_split_unweighted_nobag_kernels.h
//
// DESCRIPTION:
//   Host function for no-bag (sequence) embedding backward pass without pooling.
//   Sorts indices via transpose_embedding_input to group identical embedding
//   lookups together (run-length encoding). Then:
//     1. Classifies runs into short vs. long via the find-long-segments kernel.
//     2. Launches the CTA-per-row kernel for long runs (launched first for
//        pipelining, matching CUDA source order).
//     3. Launches the warp-per-row kernel for short runs.
{%- if not dense %}
//   Uses rowwise Adagrad for optimizer updates. Supports stochastic rounding,
//   LXU cache lookups for UVM-managed tables, and L2 / decoupled weight decay.
{%- else %}
//   Dense path: no optimizer. Accumulates gradients into grad_dev_weights,
//   which is returned as the output gradient tensor.
{%- endif %}
//
////////////////////////////////////////////////////////////////////////////////

{%  if dense %}

#include "gen_embedding_backward_dense_unweighted_nobag_kernels.h"

using Tensor = at::Tensor;
{%  else %}

#include "gen_embedding_backward_rowwise_adagrad_split_unweighted_nobag_kernels.h"

{%  endif %}

namespace fbgemm_xpu {

namespace {
class SplitEmbeddingBackwardFindLongSegments{{ find_segs_kernel }}Kernel;
}

#define DISPATCH_PLACEHOLDER_TYPES(NAME, ...) \
    return __VA_ARGS__();

    Tensor split_embedding_nobag_backward_codegen_{{ host_fn }}_unweighted_exact_xpu(
        const Tensor& grad_output,
        const Tensor& dev_weights,
        {%  if not dense %}
        const Tensor& uvm_weights,
        const Tensor& lxu_cache_weights,
        const Tensor& weights_placements,
        {%  endif %}
        const Tensor& weights_offsets,
        const c10::SymInt D_,
        const Tensor& hash_size_cumsum,
        const int64_t total_hash_size_bits,
        const Tensor& indices,
        const Tensor& offsets,
        {%  if not dense %}
        const Tensor& lxu_cache_locations,
        {%  endif %}
        const int64_t unused_,
        const int64_t max_segment_length_per_warp,
        {%  if not dense %}
        const bool stochastic_rounding,
        const int64_t info_B_num_bits, // int32_t
        const int64_t info_B_mask_int64, // uint32_t
        const bool use_uniq_cache_locations,
        const bool use_homogeneous_placements,
        Tensor momentum1_dev,
        Tensor momentum1_uvm,
        Tensor momentum1_placements,
        Tensor momentum1_offsets,
        Tensor learning_rate_tensor,
        double eps,
        double weight_decay,
        int64_t weight_decay_mode,
        double max_norm
        {%  else %}
        double unused
        {%  endif %}
    ) {
        const int64_t D = D_.guard_int(__FILE__, __LINE__);
        {%  if not dense %}
        // convert `learning rate` to float since `learning rate` is float in kernels
        TORCH_CHECK(learning_rate_tensor.is_cpu(), "learning_rate_tensor tensor needs to be on CPU. Ensure learning_rate_tensor is on CPU or contact FBGEMM team if you get this error.")
        const float learning_rate = learning_rate_tensor.item<float>();
        {%  endif %}

        TENSORS_ON_SAME_SYCL_XPU_IF_NOT_OPTIONAL(
            dev_weights,
            {%  if not dense %}
            uvm_weights,
            lxu_cache_weights,
            weights_placements,
            {%  endif %}
            weights_offsets,
            hash_size_cumsum,
            indices,
            offsets,
            {%  if not dense %}
            lxu_cache_locations,
            {%  endif %}
            grad_output);

        auto aligned_grad_output = aligned_grad_output_tensor_for_xpu_backwards(grad_output);

        SYCL_DEVICE_GUARD(dev_weights);
        auto max_D = D;
        TORCH_CHECK_LE(max_D, 2048);
        // Set total_unique_indices to total num indices by default
        const auto total_unique_indices = indices.numel();
        {%  if dense %}
        auto grad_dev_weights = zeros_like(dev_weights);
        {%  endif %}

        // short-circuit if there are zero indices.
        if (indices.numel() == 0) {
            {%  if dense %}
            return grad_dev_weights;
            {%  else %}
            return Tensor();
            {%  endif %}
        }
        int32_t T = weights_offsets.numel();

        TORCH_CHECK_GT(T, 0);
        // offsets = [B x T  + 1]
        const auto total_B = offsets.size(0) - 1;
        TORCH_CHECK_GT(total_B, 0);
        {%  if dense %}
        int32_t info_B_num_bits;
        uint32_t info_B_mask;
        std::tie(info_B_num_bits, info_B_mask) = adjust_info_B_num_bits(total_B / T, T);
        {%  else %}
        // Cast info_B_mask from int64_t to uint32_t
        const uint32_t info_B_mask = info_B_mask_int64;
        {%  endif %}

        {%  if dense %}
        auto device = c10::xpu::getCurrentXPUStream().queue().get_device();
        {%  endif %}
        int max_shared_bytes = 64 << 10;

        int shared_kb = max_shared_bytes >> 10;

        int used_shared_kb = shared_kb;

        const int used_shared_bytes = used_shared_kb << 10;

        Tensor linear_indices, linear_indices_sorted, infos_sorted,
            sorted_linear_indices_run, sorted_linear_indices_run_lengths,
            sorted_linear_indices_num_runs,
            sorted_linear_indices_cumulative_run_lengths;
        std::tie(
            linear_indices,
            linear_indices_sorted,
            infos_sorted,
            sorted_linear_indices_run,
            sorted_linear_indices_run_lengths,
            sorted_linear_indices_num_runs,
            sorted_linear_indices_cumulative_run_lengths) =
            transpose_embedding_input(
                hash_size_cumsum,
                total_hash_size_bits,
                indices,
                offsets,
                true,
                std::optional<Tensor>(),
                info_B_num_bits,
                info_B_mask,
                total_unique_indices,
                false // is_index_select
            );
        {%  if not dense %}
        Tensor lxu_cache_locations_sorted = lxu_cache_locations;
        Tensor table_unique_indices_offsets;

        if (lxu_cache_locations.size(0) > 0) {
            lxu_cache_locations_sorted = at::empty_like(lxu_cache_locations);
            // size_t temp_storage_bytes = 0;
            AT_DISPATCH_INDEX_TYPES(indices.scalar_type(), "split_embedding_nobag_backward_codegen_{{ host_fn }}_unweighted_exact_xpu_1", [&] {
                auto sorted = at::sort(linear_indices, 0, false);
                linear_indices_sorted.copy_(std::get<0>(sorted));
                auto permutation = std::get<1>(sorted);
                lxu_cache_locations_sorted.copy_(lxu_cache_locations.index_select(0, permutation));
            });
        }

        table_unique_indices_offsets = at::zeros_like(weights_placements);
        {%  endif %}

        AT_DISPATCH_INDEX_TYPES(indices.scalar_type(), "split_embedding_nobag_backward_codegen_{{ host_fn }}_unweighted_exact_xpu_2", [&] {
        DISPATCH_EMB_GRAD_CACHE_TYPES(
            dev_weights.scalar_type(),
            aligned_grad_output.scalar_type(),
            {%  if dense %}
            dev_weights.scalar_type(),
            {%  else %}
            lxu_cache_weights.scalar_type(),
            {%  endif %}
                "split_embedding_nobag_backward_codegen_{{ host_fn }}_unweighted_exact_xpu",
            [&] {

                // early memory release
                linear_indices.reset();
                linear_indices_sorted.reset();
                const auto grad_output_reshaped = aligned_grad_output;

                auto grad_output_accessor = grad_output_reshaped.packed_accessor64<grad_t, 2, RestrictPtrTraits>();

                {%  if not dense %}
                PhiloxXpuState rng_engine_inputs{};
                if (stochastic_rounding && !std::is_same_v<emb_t, float>) {
                    auto gen = at::xpu::detail::getDefaultXPUGenerator(); // XPU default generator
                    std::lock_guard<std::mutex> lock(gen.mutex());
                    auto* xpu_gen = at::check_generator<at::XPUGeneratorImpl>(gen);
                    auto [seed, offset] = xpu_gen->philox_engine_inputs(4); // reserve 4 randoms/thread unit
                    rng_engine_inputs = {seed, offset};
                }
                {%  endif %}

                DISPATCH_OPTIMAL_KERNEL(max_D, [&] {

                    auto long_run_ids = at::empty({indices.numel()}, sorted_linear_indices_run_lengths.options());
                    auto num_long_run_ids = at::zeros({1}, indices.options().dtype(at::kInt));

                    const bool use_deterministic_algorithms = at::globalContext().deterministicAlgorithms();
                    const int max_segment_length_per_cta = use_deterministic_algorithms ? INT_MAX : 1024;

                    Tensor long_run_id_to_really_long_run_ids;
                    if (use_deterministic_algorithms) {
                        long_run_id_to_really_long_run_ids =
                            at::empty(0, sorted_linear_indices_run_lengths.options());
                    } else {
                        long_run_id_to_really_long_run_ids =
                            at::empty({indices.numel()}, sorted_linear_indices_run_lengths.options());
                    }


                    auto num_really_long_run_ids = at::zeros({1}, indices.options().dtype(at::kInt));
                    auto grad_accum_counter = at::empty(
                        use_deterministic_algorithms ? 0 : (indices.numel() / max_segment_length_per_cta),
                        indices.options().dtype(at::kInt));

                    sycl::queue& queue = c10::xpu::getCurrentXPUStream().queue();
                    size_t local_range = kMaxThreads;
                    size_t global_range = div_round_up(static_cast<size_t>(total_unique_indices), local_range) * local_range;

                    // Step 1: Classify runs into short (warp-per-row) and long (CTA-per-row).
                    // Fills long_run_ids with the run IDs exceeding max_segment_length_per_warp.
                    queue.submit([&](sycl::handler& cgh) {
                        cgh.parallel_for<SplitEmbeddingBackwardFindLongSegments{{ find_segs_kernel }}Kernel>(
                            sycl::nd_range<1>(
                                global_range,
                                local_range),
                            SplitEmbeddingBackwardFindLongSegments(
                                sorted_linear_indices_num_runs.packed_accessor32<int32_t, 1, RestrictPtrTraits>(),
                                sorted_linear_indices_run_lengths.packed_accessor32<int32_t, 1, RestrictPtrTraits>(),
                                long_run_ids.packed_accessor32<int32_t, 1, RestrictPtrTraits>(),
                                num_long_run_ids.packed_accessor32<int32_t, 1, RestrictPtrTraits>(),
                                long_run_id_to_really_long_run_ids.packed_accessor32<int32_t, 1, RestrictPtrTraits>(),
                                num_really_long_run_ids.packed_accessor32<int32_t, 1, RestrictPtrTraits>(),
                                grad_accum_counter.packed_accessor32<int32_t, 1, RestrictPtrTraits>(),
                                max_segment_length_per_warp,
                                max_segment_length_per_cta,
                                use_deterministic_algorithms));
                    });

                    // A temp buffer to accumulate gradients with atomics.
                    auto temp_grad_accum = at::zeros(
                        {use_deterministic_algorithms ? 0 : grad_accum_counter.numel(), max_D},
                        aligned_grad_output.options().dtype(std::is_same<cache_t, double>::value ? at::kDouble : at::kFloat));

                    {%  if not dense %}
                    DISPATCH_PLACEHOLDER_TYPES(
                    "split_embedding_backward_rowwise_adagrad_exact_placeholder_type_kernel",
                    [&] {

                    {%  endif %}
                        // Step 2: CTA-per-row kernel — long segments (SL >= max_segment_length_per_warp).
                        // CUDA equivalent: split_embedding_nobag_backward_codegen_{{ "dense" if dense else "rowwise_adagrad" }}_unweighted_kernel_cta_per_row_1
                        constexpr auto kCacheAccBytes = sizeof(at::acc_type<cache_t, true>);

                        int32_t num_cta_per_row_groups = kMaxThreads / kThreadGroupSize;

                        validate_local_mem_size(queue, used_shared_bytes);
                        const size_t cta_per_row_smem_bytes = compute_num_groups_and_dynamic_smem_bytes(
                            &num_cta_per_row_groups,
                            [&] (int num_groups) {
                                return num_groups * kCacheAccBytes * 4 * kThreadGroupSize * max_vecs_per_thread;
                            },
                            used_shared_bytes
                        );

                        const int32_t cta_per_row_grid_size = std::min<int32_t>(
                            div_round_up(static_cast<int32_t>(total_unique_indices), static_cast<int32_t>(kMaxThreads)),
                            static_cast<int32_t>(get_max_work_groups_()));

                        {%  if dense %}
                        const size_t cta_local_x = kThreadGroupSize;
                        const size_t cta_local_y = num_cta_per_row_groups;
                        const size_t cta_grid = cta_per_row_grid_size;
                        {%  else %}
                        const size_t c_local_x = kThreadGroupSize;
                        const size_t c_local_y = num_cta_per_row_groups;
                        const size_t c_grid = cta_per_row_grid_size;
                        {%  endif %}
                        queue.submit([&](sycl::handler& cgh) {
                            // Allocate local memory (equivalent to shared memory)
                            auto smem = sycl::local_accessor<cache_t, 1>(
                                cta_per_row_smem_bytes / sizeof(cache_t),
                                cgh);
                            cgh.parallel_for<SplitEmbeddingNobagBackwardCodegen{{ optimizer }}UnweightedKernelCtaPerRow<emb_t, grad_t, cache_t, index_t, kFixedMaxVecsPerThread, kThreadGroupSize, kUseVecBlocking>>(
                                sycl::nd_range<2>(
                                    sycl::range<2>(
                                        {%  if dense %}
                                        cta_grid * cta_local_y, cta_local_x),
                                    sycl::range<2>(
                                        cta_local_y, cta_local_x)
                                        {%  else %}
                                        c_grid * c_local_y, c_local_x),
                                    sycl::range<2>(
                                        c_local_y, c_local_x)
                                        {%  endif %}
                                ),
                                SplitEmbeddingNobagBackwardCodegen{{ optimizer }}UnweightedKernelCtaPerRow<emb_t, grad_t, cache_t, index_t, kFixedMaxVecsPerThread, kThreadGroupSize, kUseVecBlocking>(
                                    grad_output_accessor,
                                    dev_weights.packed_accessor64<emb_t, 1, RestrictPtrTraits>(),
                                    {%  if not dense %}
                                    uvm_weights.packed_accessor64<emb_t, 1, RestrictPtrTraits>(),
                                    lxu_cache_weights.packed_accessor64<cache_t, 2, RestrictPtrTraits>(),
                                    weights_placements.packed_accessor32<int32_t, 1, RestrictPtrTraits>(), // if optimizer != "none"
                                    {%  endif %}
                                    weights_offsets.packed_accessor32<int64_t, 1, RestrictPtrTraits>(),
                                    D,
                                    hash_size_cumsum.packed_accessor32<int64_t, 1, RestrictPtrTraits>(),
                                    sorted_linear_indices_run.packed_accessor32<index_t, 1, RestrictPtrTraits>(),
                                    sorted_linear_indices_cumulative_run_lengths.packed_accessor32<int32_t, 1, RestrictPtrTraits>(),
                                    long_run_ids.packed_accessor32<int32_t, 1, RestrictPtrTraits>(),
                                    num_long_run_ids.packed_accessor32<int32_t, 1, RestrictPtrTraits>(),
                                    infos_sorted.packed_accessor32<int64_t, 1, RestrictPtrTraits>(),
                                    {%  if not dense %}
                                    lxu_cache_locations_sorted.packed_accessor32<int32_t, 1, RestrictPtrTraits>(),
                                    use_uniq_cache_locations,
                                    table_unique_indices_offsets.packed_accessor32<int32_t, 1, RestrictPtrTraits>(),
                                    stochastic_rounding,
                                    rng_engine_inputs, // if not dense and optimizer != "none"
                                    {%  endif %}
                                    {%  if dense %}
                                    grad_dev_weights.packed_accessor64<emb_t, 1, RestrictPtrTraits>(),
                                    {%  endif %}
                                    long_run_id_to_really_long_run_ids.packed_accessor32<int32_t, 1, RestrictPtrTraits>(),
                                    temp_grad_accum.packed_accessor32<at::acc_type<cache_t, true>, 2, RestrictPtrTraits>(),
                                    grad_accum_counter.packed_accessor32<int32_t, 1, RestrictPtrTraits>(),
                                    max_segment_length_per_cta,
                                    use_deterministic_algorithms,
                                    max_vecs_per_thread,
                                    {%  if not dense %}
                                    momentum1_dev.packed_accessor64<at::acc_type<cache_t, true>, 1, RestrictPtrTraits>(),
                                    momentum1_uvm.packed_accessor64<at::acc_type<cache_t, true>, 1, RestrictPtrTraits>(),
                                    momentum1_placements.packed_accessor32<int32_t, 1, RestrictPtrTraits>(),
                                    momentum1_offsets.packed_accessor32<int64_t, 1, RestrictPtrTraits>(),
                                    {%  endif %}
                                    smem{%  if dense %},
                                    unused{%  else %},
                                    learning_rate,
                                    eps,
                                    weight_decay,
                                    weight_decay_mode,
                                    max_norm{%  endif %}
                                )
                            );
                        });

                        // Step 3: Warp-per-row kernel — short segments (SL < max_segment_length_per_warp).
                        // CUDA equivalent: split_embedding_nobag_backward_codegen_{{ "dense" if dense else "rowwise_adagrad" }}_unweighted_kernel_warp_per_row_1
                        int32_t num_warp_per_row_groups = kBackwardMaxThreads / kThreadGroupSize;
                        int32_t warp_per_row_smem_bytes = 0;
                        if constexpr (kUseVecBlocking) {
                            warp_per_row_smem_bytes = compute_num_groups_and_dynamic_smem_bytes(
                                &num_warp_per_row_groups,
                                [&] (int num_groups) {
                                    return num_groups * kCacheAccBytes * max_D;
                                },
                                used_shared_bytes
                            );
                        }

                        const int32_t warp_per_row_grid_size = std::min<int32_t>(
                            div_round_up(static_cast<int32_t>(total_unique_indices), static_cast<int32_t>(num_warp_per_row_groups)),
                            static_cast<int32_t>(get_max_work_groups_()));

                        {%  if dense %}
                        const size_t warp_local_x = kThreadGroupSize;
                        const size_t warp_local_y = num_warp_per_row_groups;
                        const size_t warp_grid = warp_per_row_grid_size;
                        {%  else %}
                        const size_t w_local_x = kThreadGroupSize;
                        const size_t w_local_y = num_warp_per_row_groups;
                        const size_t w_grid = warp_per_row_grid_size;
                        {%  endif %}
                        queue.submit([&](sycl::handler& cgh) {
                            sycl::local_accessor<cache_t, 1> smem(
                                kUseVecBlocking ? (warp_per_row_smem_bytes / sizeof(cache_t)) : 0,
                                cgh);
                            cgh.parallel_for<SplitEmbeddingNobagBackwardCodegen{{ optimizer }}UnweightedKernelWarpPerRow<emb_t, grad_t, cache_t, index_t, kFixedMaxVecsPerThread, kThreadGroupSize, kUseVecBlocking>>(
                                sycl::nd_range<2>(
                                    sycl::range<2>(
                                        {%  if dense %}
                                        warp_grid * warp_local_y,
                                        warp_local_x),
                                    sycl::range<2>(
                                        warp_local_y,
                                        warp_local_x)
                                        {%  else %}
                                        w_grid * w_local_y,
                                        w_local_x),
                                    sycl::range<2>(
                                        w_local_y,
                                        w_local_x)
                                        {%  endif %}
                                ),
                                SplitEmbeddingNobagBackwardCodegen{{ optimizer }}UnweightedKernelWarpPerRow<emb_t, grad_t, cache_t, index_t, kFixedMaxVecsPerThread, kThreadGroupSize, kUseVecBlocking>(
                                    grad_output_accessor,
                                    dev_weights.packed_accessor64<emb_t, 1, RestrictPtrTraits>(),
                                    {%  if not dense %}
                                    uvm_weights.packed_accessor64<emb_t, 1, RestrictPtrTraits>(),
                                    lxu_cache_weights.packed_accessor64<cache_t, 2, RestrictPtrTraits>(),
                                    weights_placements.packed_accessor32<int32_t, 1, RestrictPtrTraits>(), // if optimizer != "none"
                                    {%  endif %}
                                    weights_offsets.packed_accessor32<int64_t, 1, RestrictPtrTraits>(),
                                    D,
                                    hash_size_cumsum.packed_accessor32<int64_t, 1, RestrictPtrTraits>(),
                                    sorted_linear_indices_run.packed_accessor32<index_t, 1, RestrictPtrTraits>(),
                                    sorted_linear_indices_cumulative_run_lengths.packed_accessor32<int32_t, 1, RestrictPtrTraits>(),
                                    infos_sorted.packed_accessor32<int64_t, 1, RestrictPtrTraits>(),
                                    {%  if not dense %}
                                    lxu_cache_locations_sorted.packed_accessor32<int32_t, 1, RestrictPtrTraits>(),
                                    use_uniq_cache_locations,
                                    table_unique_indices_offsets.packed_accessor32<int32_t, 1, RestrictPtrTraits>(),
                                    {%  endif %}
                                    sorted_linear_indices_num_runs.packed_accessor32<int32_t, 1, RestrictPtrTraits>(),
                                    {%  if dense %}
                                    static_cast<int32_t>(max_segment_length_per_warp),
                                    grad_dev_weights.packed_accessor64<emb_t, 1, RestrictPtrTraits>(),
                                    {%  else %}
                                    max_segment_length_per_warp,
                                    stochastic_rounding,
                                    rng_engine_inputs, // if not dense and optimizer != "none"
                                    {%  endif %}
                                    max_D,
                                    max_vecs_per_thread,
                                    {%  if not dense %}
                                    momentum1_dev.packed_accessor64<at::acc_type<cache_t, true>, 1, RestrictPtrTraits>(),
                                    momentum1_uvm.packed_accessor64<at::acc_type<cache_t, true>, 1, RestrictPtrTraits>(),
                                    momentum1_placements.packed_accessor32<int32_t, 1, RestrictPtrTraits>(),
                                    momentum1_offsets.packed_accessor32<int64_t, 1, RestrictPtrTraits>(),
                                    {%  endif %}
                                    smem{%  if dense %},
                                    unused{%  else %},
                                    learning_rate,
                                    eps,
                                    weight_decay,
                                    weight_decay_mode,
                                    max_norm{%  endif %}
                                )
                            );
                        });

                    {%  if not dense %}
                    }); // DISPATCH_PLACEHOLDER_TYPES
                    {%  endif %}

                }); // DISPATCH_OPTIMAL_KERNEL
        }); // DISPATCH_EMB_GRAD_CACHE_TYPES
        }); // AT_DISPATCH_INDEX_TYPES
        {%  if dense %}
        return grad_dev_weights;
        {%  else %}
        return Tensor();
        {%  endif %}
    }
} // namespace fbgemm_xpu

TORCH_LIBRARY_IMPL(fbgemm, XPU, m) {
    m.impl("split_embedding_nobag_backward_codegen_{{ host_fn }}_unweighted_exact_xpu", &fbgemm_xpu::split_embedding_nobag_backward_codegen_{{ host_fn }}_unweighted_exact_xpu);
}
