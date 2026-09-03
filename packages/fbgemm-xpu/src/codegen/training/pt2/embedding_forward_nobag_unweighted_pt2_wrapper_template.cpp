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

{%- set mdesc = "split" %}
{%- set optimizer = "rowwise_adagrad" %}

////////////////////////////////////////////////////////////////////////////////
// PT2 WRAPPER IMPLEMENTATION - {{ mdesc | upper }} EMBEDDING NOBAG UNWEIGHTED
////////////////////////////////////////////////////////////////////////////////
//
// CUDA SOURCE MAPPING:
//   CUDA Template: pt2/embedding_split_host_pt2_cuda_wrapper_template.cpp
//
// Codegen file output:
//   gen_embedding_forward_{{ mdesc }}_nobag_unweighted_pt2_xpu_wrapper.sycl
//   gen_embedding_backward_{{ mdesc }}_{{ optimizer }}_nobag_unweighted_pt2_xpu_wrapper.sycl
//
// DESCRIPTION:
//   PT2 (PyTorch 2.0) compilation wrapper implementation for split embedding (nobag, unweighted).
//   Uses PyTorch dispatcher to call the actual XPU kernel implementation.
//   Controlled by the `is_forward` Jinja2 variable at codegen time.
//
////////////////////////////////////////////////////////////////////////////////

#include <torch/all.h>
#include <torch/library.h>
#include <ATen/ATen.h>

using Tensor = at::Tensor;

namespace fbgemm_xpu {

{%- if is_forward %}
    ////////////////////////////////////////////////////////////////////////////////
    // {{ mdesc }}_embedding_nobag_forward_unweighted_pt2_xpu_wrapper
    ////////////////////////////////////////////////////////////////////////////////
    //
    // IMPLEMENTATION:
    //   Uses torch::Dispatcher to find and call the registered forward kernel operator.
    //   Converts parameters from PT2 wrapper format to kernel format.
    //
    ////////////////////////////////////////////////////////////////////////////////

    Tensor {{ mdesc }}_embedding_nobag_codegen_forward_unweighted_pt2_xpu_wrapper(
        const Tensor& /*host_weights*/,
        const Tensor& dev_weights,
        const Tensor& uvm_weights,
        const Tensor& lxu_cache_weights,
        const Tensor& weights_placements,
        const Tensor& weights_offsets,
        const c10::SymInt D,
        const Tensor& hash_size_cumsum,
        const Tensor& indices,
        const Tensor& offsets,
        const Tensor& lxu_cache_locations,
        const Tensor& uvm_cache_stats,
        const bool is_experimental,
        const int64_t output_dtype
        ){
        static auto op =
            torch::Dispatcher::singleton()
                .findSchemaOrThrow("fbgemm::{{ mdesc }}_embedding_nobag_forward_unweighted_xpu", "")
                .typed<Tensor(
                    const Tensor& /*host_weights*/,
                    const Tensor& /*dev_weights*/,
                    const Tensor& /*uvm_weights*/,
                    const Tensor& /*lxu_cache_weights*/,
                    const Tensor& /*weights_placements*/,
                    const c10::SymInt /*D*/,
                    const Tensor& /*indices*/,
                    const Tensor& /*offsets*/,
                    const Tensor& /*row_addrs or lxu_cache_locations*/,
                    const Tensor& /*uvm_cache_stats_*/,
                    const int64_t /*output_dtype*/,
                    const bool
                )>();

        return op.call(
                dev_weights,
                uvm_weights,
                lxu_cache_weights,
                weights_placements,
                weights_offsets,
                D,
                indices,
                offsets,
                lxu_cache_locations,
                uvm_cache_stats,
                output_dtype,
                is_experimental
        );
    }

{%- else %}

    ////////////////////////////////////////////////////////////////////////////////
    // {{ mdesc }}_embedding_nobag_backward_codegen_{{ optimizer }}_unweighted_pt2_xpu_wrapper
    ////////////////////////////////////////////////////////////////////////////////
    //
    // IMPLEMENTATION:
    //   Uses torch::Dispatcher to find and call the registered backward kernel operator.
    //   Converts parameters from PT2 wrapper format to kernel format.
    //
    ////////////////////////////////////////////////////////////////////////////////

    Tensor {{ mdesc }}_embedding_nobag_backward_codegen_{{ optimizer }}_unweighted_pt2_xpu_wrapper(
        const Tensor& grad_output,
        const Tensor& /*host_weights*/,
        const Tensor& dev_weights,
        const Tensor& uvm_weights,
        const Tensor& lxu_cache_weights,
        const Tensor& weights_placements,
        const Tensor& weights_offsets,
        const c10::SymInt D,
        const Tensor& hash_size_cumsum,
        const int64_t total_hash_size_bits,
        const Tensor& indices,
        const Tensor& offsets,
        const Tensor& lxu_cache_locations,
        const int64_t BT_block_size,
        const int64_t max_segment_length_per_warp,
        const bool stochastic_rounding,
        const int64_t info_B_num_bits,
        const int64_t info_B_mask_int64,
        const bool use_uniq_cache_locations,
        const bool use_homogeneous_placements,
        Tensor momentum1_host,
        Tensor momentum1_dev,
        Tensor momentum1_uvm,
        Tensor momentum1_placements,
        Tensor momentum1_offsets,
        Tensor learning_rate_tensor,
        double eps,
        double weight_decay,
        int64_t weight_decay_mode,
        double max_norm
        ){
        static auto op =
            torch::Dispatcher::singleton()
                .findSchemaOrThrow("fbgemm::{{ mdesc }}_embedding_nobag_backward_codegen_{{ optimizer }}_unweighted_exact_xpu", "")
                .typed<Tensor(
                        const Tensor& /*grad_output*/,
                        const Tensor& /*dev_weights*/,
                        const Tensor& /*uvm_weights*/,
                        const Tensor& /*lxu_cache_weights*/,
                        const Tensor& /*weights_placements*/,
                        const Tensor& /*weights_offsets*/,
                        const c10::SymInt /*D*/,
                        const Tensor& /*hash_size_cumsum*/,
                        const int64_t /*total_hash_size_bits*/,
                        const Tensor& /*indices*/,
                        const Tensor& /*offsets*/,
                        const Tensor& /*lxu_cache_locations*/,
                        const int64_t /*BT_block_size*/,
                        const int64_t /*max_segment_length_per_warp*/,
                        const bool /*stochastic_rounding*/,
                        const int64_t /*info_B_num_bits*/,
                        const int64_t /*info_B_mask_int64*/,
                        const bool /*use_uniq_cache_locations*/,
                        const bool /*use_homogeneous_placements*/,
                        Tensor, /*momentum1_dev*/
                        Tensor, /*momentum1_uvm*/
                        Tensor, /*momentum1_placements*/
                        Tensor, /*momentum1_offsets*/
                        Tensor, /*learning_rate_tensor*/
                        double, /*eps*/
                        double, /*weight_decay*/
                        int64_t, /*weight_decay_mode*/
                        double  /*max_norm*/
                )>();

        return op.call(
            grad_output,
            dev_weights,
            uvm_weights,
            lxu_cache_weights,
            weights_placements,
            weights_offsets,
            D,
            hash_size_cumsum,
            total_hash_size_bits,
            indices,
            offsets,
            lxu_cache_locations,
            BT_block_size,
            max_segment_length_per_warp,
            stochastic_rounding,
            info_B_num_bits,
            info_B_mask_int64,
            use_uniq_cache_locations,
            use_homogeneous_placements,
            momentum1_dev,
            momentum1_uvm,
            momentum1_placements,
            momentum1_offsets,
            learning_rate_tensor,
            eps,
            weight_decay,
            weight_decay_mode,
            max_norm
        );
    }

{%- endif %}

} // namespace fbgemm_xpu

////////////////////////////////////////////////////////////////////////////////
// TORCH LIBRARY IMPLEMENTATION REGISTRATION
////////////////////////////////////////////////////////////////////////////////
TORCH_LIBRARY_IMPL(fbgemm, XPU, m) {
{%- if is_forward %}
    m.impl("{{ mdesc }}_embedding_nobag_codegen_forward_unweighted_pt2_wrapper",
        &fbgemm_xpu::{{ mdesc }}_embedding_nobag_codegen_forward_unweighted_pt2_xpu_wrapper);
{%- else %}
    m.impl("{{ mdesc }}_embedding_nobag_backward_codegen_{{ optimizer }}_unweighted_pt2_wrapper",
        &fbgemm_xpu::{{ mdesc }}_embedding_nobag_backward_codegen_{{ optimizer }}_unweighted_pt2_xpu_wrapper);
{%- endif %}
}
