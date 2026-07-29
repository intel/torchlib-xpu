/*
 * Copyright (c) Meta Platforms, Inc. and affiliates. All rights reserved.
 * Copyright (c) 2026 Intel Corporation. All Rights Reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */

////////////////////////////////////////////////////////////////////////////////
// SYCL PORT MAPPING TO FBGEMM CUDA SOURCE
////////////////////////////////////////////////////////////////////////////////
//
// This file is a SYCL port of the FBGEMM dense embedding lookup operators.
//
// ORIGINAL CUDA SOURCE:
//   Template Path:
//   FBGEMM/fbgemm_gpu/codegen/training/backward/embedding_backward_split_host_template.cpp
//   Generated File:
//   fbgemm_gpu/_skbuild/linux-x86_64-3.10/cmake-build/gen_embedding_backward_split_dense.cpp
//
// ARCHITECTURE OVERVIEW:
//   - Main entry: split_embedding_lookup_dense_function (Host Wrapper)
//   - Route 1: VBE (Variable Batch Embedding) - B_offsets present [NOT
//   IMPLEMENTED]
//   - Route 2: No-Bag (pooling_mode == NONE) [IMPLEMENTED]
//   - Route 3: Pooled Standard (SUM/MEAN pooling) [NOT IMPLEMENTED]
//
// IMPLEMENTATION STATUS:
//   ✅ SplitNoBagLookupFunctionDenseOpXPU - No-bag forward/backward
//   ❌ SplitVBELookupFunctionDenseOpXPU - VBE route (not implemented)
//   ❌ SplitLookupFunctionDenseOpXPU - Pooled route (not implemented)
//
////////////////////////////////////////////////////////////////////////////////

#include <ATen/Operators.h>
#include <c10/xpu/XPUStream.h>
#include <torch/all.h>
#include <torch/csrc/autograd/record_function_ops.h>
#include <torch/library.h>

#include <string>
#include <utility>

#include <sycl/sycl.hpp>

#include "fbgemm_utils/feature_gates.h"
#include "fbgemm_utils/utils.h"


using Tensor = at::Tensor;
namespace profiler = torch::autograd::profiler;

namespace fbgemm_xpu {

Tensor split_embedding_nobag_backward_codegen_dense_unweighted_exact_xpu(
    const Tensor& grad_output, const Tensor& dev_weights,
    const Tensor& weights_offsets, const c10::SymInt D,
    const Tensor& hash_size_cumsum, const int64_t total_hash_size_bits,
    const Tensor& indices, const Tensor& offsets, const int64_t BT_block_size,
    const int64_t max_segment_length_per_warp, double unused = 0);

Tensor dense_embedding_nobag_forward_unweighted_xpu(
    const Tensor& dev_weights, const Tensor& weights_offsets,
    const c10::SymInt D, const Tensor& indices, const Tensor& offsets,
    const int64_t output_dtype, const bool is_experimental);

////////////////////////////////////////////////////////////////////////////////
// SplitNoBagLookupFunctionDenseOpXPU - No-Bag (Sequence) Embedding Autograd
////////////////////////////////////////////////////////////////////////////////
//
// CUDA SOURCE MAPPING:
//   CUDA Class: SplitNoBagLookupFunction_dense_Op
//   CUDA File: gen_embedding_backward_split_dense.cpp
//   CUDA Path: FBGEMM/fbgemm_gpu/_skbuild/linux-x86_64-3.10/cmake-build/
//
// DESCRIPTION:
//   Autograd function for no-bag (sequence) embedding lookups.
//   Used when pooling_mode == PoolingMode::NONE.
//   Returns raw embedding vectors without pooling.
//
// FORWARD PATH:
//   Calls: dense_embedding_nobag_forward_unweighted_xpu
//   CUDA Equivalent: dense_embedding_nobag_forward_codegen_unweighted_cuda
//
// BACKWARD PATH:
//   Calls: split_embedding_nobag_backward_codegen_dense_unweighted_exact_xpu
//   CUDA Equivalent:
//   split_embedding_nobag_backward_codegen_dense_unweighted_exact_cuda
//
////////////////////////////////////////////////////////////////////////////////
class SplitNoBagLookupFunctionDenseOpXPU
    : public torch::autograd::Function<SplitNoBagLookupFunctionDenseOpXPU> {
 public:
    static constexpr bool is_traceable = true;

    static torch::autograd::variable_list
    forward(torch::autograd::AutogradContext* ctx, const int64_t output_dtype,
            const Tensor& dev_weights, const Tensor& weights_offsets,
            const c10::SymInt D, const Tensor& hash_size_cumsum,
            const int64_t total_hash_size_bits, const Tensor& indices,
            const Tensor& offsets) {
        const auto T = weights_offsets.sym_numel();
        const auto max_B_ = offsets.sym_size(0) / T;

        // Annotate Kineto trace
        static const bool is_annotate_trace_enabled =
            config::is_feature_enabled(
                config::FeatureGateName::TBE_ANNOTATE_KINETO_TRACE);
        std::string op_annotation = "";
        c10::intrusive_ptr<profiler::PythonRecordFunction> record_trace;
        if (is_annotate_trace_enabled) {
            std::stringstream ss;
            ss << "[" << "weighted=F," << "pooled=F," << "vbe=F,"
               << "avg_B=" << (max_B_) << "," << "max_B=" << max_B_ << ","
               << "T=" << T << "," << "avg_D=" << (D) << "," << "max_D=" << D
               << "," << "num_indices=" << indices.sym_numel() << ","
               << "avg_pooling_fac="
               << (static_cast<c10::SymFloat>(indices.sym_numel()) / T / max_B_)
               << "]";
            op_annotation = ss.str();
            record_trace = profiler::record_function_enter_new("dense_tbe_fwd" +
                                                               op_annotation);
            ctx->saved_data["op_annotation"] = op_annotation;
        }

        int32_t info_B_num_bits = kDefaultInfoBNumBits;
        uint32_t info_B_mask = (1u << info_B_num_bits) - 1;
        if (max_B_.is_symbolic()) {
            info_B_num_bits = 22;
            info_B_mask = (1u << info_B_num_bits) - 1;
        } else {
            std::tie(info_B_num_bits, info_B_mask) =
                adjust_info_B_num_bits(max_B_.guard_int(__FILE__, __LINE__),
                                       T.guard_int(__FILE__, __LINE__));
        }

        ctx->save_for_backward({
            dev_weights,
            weights_offsets,
            hash_size_cumsum,
            indices,
            offsets,
        });
        ctx->saved_data["D"] = D;
        ctx->saved_data["total_hash_size_bits"] = total_hash_size_bits;
        ctx->saved_data["info_B_num_bits"] = info_B_num_bits;
        const auto info_B_mask_int64 = static_cast<int64_t>(info_B_mask);
        ctx->saved_data["info_B_mask"] = info_B_mask_int64;
        const auto& flatten_dev_weights = dev_weights;

        static auto embedding_forward_op =
            torch::Dispatcher::singleton()
                .findSchemaOrThrow(
                    "fbgemm::dense_embedding_nobag_forward_unweighted_xpu", "")
                .typed<
                    decltype(dense_embedding_nobag_forward_unweighted_xpu)>();

        auto output =
            embedding_forward_op.call(flatten_dev_weights, weights_offsets, D,
                                      indices, offsets, output_dtype, false);

        if (is_annotate_trace_enabled) {
            record_trace->record.end();
        }
        using torch::autograd::Variable;
        return {output};
    }

    static torch::autograd::variable_list
    backward(torch::autograd::AutogradContext* ctx,
             torch::autograd::variable_list grad_outputs) {
        const auto saved = ctx->get_saved_variables();
        auto savedItr = std::begin(saved);
        auto dev_weights = *savedItr++;
        auto weights_offsets = *savedItr++;
        auto hash_size_cumsum = *savedItr++;
        auto indices = *savedItr++;
        auto offsets = *savedItr++;
        auto D = ctx->saved_data["D"].toSymInt();
        auto total_hash_size_bits =
            ctx->saved_data["total_hash_size_bits"].toInt();
        [[maybe_unused]] const int32_t info_B_num_bits =
            ctx->saved_data["info_B_num_bits"].toInt();
        [[maybe_unused]] const int64_t info_B_mask_int64 =
            ctx->saved_data["info_B_mask"].toInt();

        static const bool is_annotate_trace_enabled =
            config::is_feature_enabled(
                config::FeatureGateName::TBE_ANNOTATE_KINETO_TRACE);
        c10::intrusive_ptr<profiler::PythonRecordFunction> record_trace;
        if (is_annotate_trace_enabled) {
            auto& op_annotation =
                ctx->saved_data["op_annotation"].toStringRef();
            record_trace = profiler::record_function_enter_new("split_tbe_bwd" +
                                                               op_annotation);
        }

        TORCH_CHECK_EQ(grad_outputs.size(), 1);

        constexpr int32_t BT_block_size = 32;
        constexpr int32_t max_segment_length_per_warp = 32;

        using torch::autograd::Variable;
        auto& grad_output = grad_outputs[0];
        Tensor grad_dev_weights;

        static auto embedding__unweighted_backward_op =
            torch::Dispatcher::singleton()
                .findSchemaOrThrow(
                    "fbgemm::split_embedding_nobag_backward_codegen_dense_"
                    "unweighted_exact_xpu",
                    "")
                .typed<
                    decltype(split_embedding_nobag_backward_codegen_dense_unweighted_exact_xpu)>();

        grad_dev_weights = embedding__unweighted_backward_op.call(
            grad_output, dev_weights, weights_offsets, D, hash_size_cumsum,
            total_hash_size_bits, indices, offsets, BT_block_size,
            max_segment_length_per_warp,
            /*unused=*/0);

        if (is_annotate_trace_enabled) {
            record_trace->record.end();
        }

        return {Variable(),        // output_dtype
                grad_dev_weights,  // dev_weights
                Variable(),        // weights_offsets
                Variable(),        // D
                Variable(),        // hash_size_cumsum
                Variable(),        // total_hash_size_bits
                Variable(),        // indices
                Variable(),        // offsets
                Variable()};
    }
};

////////////////////////////////////////////////////////////////////////////////
// split_embedding_lookup_dense_function_xpu - Main Entry Point
////////////////////////////////////////////////////////////////////////////////
//
// CUDA SOURCE MAPPING:
//   CUDA Function: split_embedding_lookup_dense_function
//   CUDA File: gen_embedding_backward_split_dense.cpp
//   CUDA Path: FBGEMM/fbgemm_gpu/_skbuild/linux-x86_64-3.10/cmake-build/
//
// DESCRIPTION:
//   Main host wrapper for dense embedding lookup operations.
//   Dispatches to different autograd functions based on input parameters.
//
// DISPATCH LOGIC (from CUDA):
//   if (B_offsets.has_value())                    → VBE Route [NOT IMPLEMENTED]
//   else if (pooling_mode == PoolingMode::NONE)  → No-Bag Route [IMPLEMENTED]
//   else                                          → Pooled Route [NOT
//   IMPLEMENTED]
//
// OPERATOR REGISTRATION:
//   PyTorch Op: fbgemm::dense_embedding_codegen_lookup_function
//   Device Key: AutogradXPU
//
////////////////////////////////////////////////////////////////////////////////
at::Tensor split_embedding_lookup_dense_function_xpu(
    const Tensor& dev_weights, const Tensor& weights_offsets,
    const Tensor& D_offsets, const c10::SymInt total_D, const c10::SymInt max_D,
    const Tensor& hash_size_cumsum, const int64_t total_hash_size_bits,
    const Tensor& indices, const Tensor& offsets, const int64_t pooling_mode,
    const std::optional<Tensor>& indice_weights,
    const std::optional<Tensor>& feature_requires_grad,
    const int64_t output_dtype = static_cast<int64_t>(SparseType::FP32),
    const std::optional<Tensor>& B_offsets = std::nullopt,
    const std::optional<Tensor>& vbe_output_offsets_feature_rank = std::nullopt,
    const std::optional<Tensor>& vbe_B_offsets_rank_per_feature = std::nullopt,
    const c10::SymInt max_B = -1, const c10::SymInt max_B_feature_rank = -1,
    const c10::SymInt vbe_output_size = -1, const bool mixed_D = true) {
    if (B_offsets.has_value()) {
        TORCH_CHECK(false,
                    "dense_embedding_codegen_lookup_function with B_offsets is "
                    "not implemented yet.");

    } else if (static_cast<PoolingMode>(pooling_mode) == PoolingMode::NONE) {
        // no bag

        return SplitNoBagLookupFunctionDenseOpXPU::apply(
            output_dtype, dev_weights, weights_offsets, max_D, hash_size_cumsum,
            total_hash_size_bits, indices, offsets)[0];

    } else {
        TORCH_CHECK(false,
                    "dense_embedding_codegen_lookup_function is not "
                    "implemented yet for SplitLookupFunction_dense_Op.");
    }
}

TORCH_LIBRARY_IMPL(fbgemm, AutogradXPU, m) {
    m.impl("dense_embedding_codegen_lookup_function",
           &split_embedding_lookup_dense_function_xpu);
}

}  // namespace fbgemm_xpu
