/*
 * Copyright (c) Meta Platforms, Inc. and affiliates. All rights reserved.
 * Copyright (c) 2026 Intel Corporation. All Rights Reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */

////////////////////////////////////////////////////////////////////////////////
// SYCL PORT MAPPING TO FBGEMM CUDA SOURCE
////////////////////////////////////////////////////////////////////////////////
//
// This file contains SYCL implementations of FBGEMM 2D sparse data permutation
// kernels and their host orchestration.
//
// ORIGINAL CUDA SOURCE:
//   File: fbgemm_gpu/src/sparse_ops/sparse_permute_2d.cu
//
// KERNEL IMPLEMENTATIONS:
//   Permute2DLengthsKernel::operator()
//     → permute_2D_lengths_kernel (CUDA)
//
//   Permute2DDataKernel::operator()
//     → permute_2D_data_kernel<has_weight, ...> (CUDA)
//
// HOST FUNCTION:
//   permute_2D_sparse_data_xpu
//     → permute_2D_sparse_data_cuda (CUDA)
//
//   permute_2D_sparse_preallocated_out_xpu
//     → permute_2D_sparse_preallocated_out_cuda (CUDA)
//
////////////////////////////////////////////////////////////////////////////////

#include "permute_2d_sparse_data.h"
#include "sparse_async_cumsum.h"

namespace fbgemm_xpu {

// ============================================================================
// SYCL Kernel Functors - Operator Implementations
// ============================================================================

/**
 * @brief Permute2DLengthsKernel operator implementation.
 *
 * Uses a grid-stride loop over the flattened [T, B] index space so the same
 * launch geometry works for any T * B. Row-major layout is assumed so that
 * the linear index b_t maps to (t = b_t / B, b = b_t % B).
 */
template <typename index_t>
void Permute2DLengthsKernel<index_t>::operator()(
    const sycl::nd_item<1>& item) const {
    const int64_t global_id = item.get_global_id(0);
    const int64_t global_range = item.get_global_range(0);
    const int64_t total = static_cast<int64_t>(T_) * static_cast<int64_t>(B_);

    for (int64_t b_t = global_id; b_t < total; b_t += global_range) {
        const int32_t b = static_cast<int32_t>(b_t % B_);
        const int32_t t = static_cast<int32_t>(b_t / B_);
        permuted_lengths_[b_t] = lengths_[permute_[t] * B_ + b];
    }
}

/**
 * @brief Permute2DDataKernel operator implementation.
 *
 * Copies indices for each (t, b) segment; when `has_weight` is true it also
 * copies the associated weights row (all `weights_columns_` columns). The
 * weight-copy branch is a compile-time `if constexpr`, so the no-weights
 * instantiation pays no runtime cost.
 */
template <
    bool has_weight,
    typename offsets_t,
    typename indices_t,
    typename weights_t>
void Permute2DDataKernel<has_weight, offsets_t, indices_t, weights_t>::operator()(
    const sycl::nd_item<2>& item) const {
    const auto b_t_start =
        item.get_group(0) * item.get_local_range(1) + item.get_local_id(1);
    const auto stride = item.get_group_range(0) * item.get_local_range(1);

    for (int32_t b_t = static_cast<int32_t>(b_t_start); b_t < B_ * T_;
         b_t += static_cast<int32_t>(stride)) {
        const int32_t b = b_t % B_;
        const int32_t t = b_t / B_;

        const offsets_t output_start = output_offsets_[b_t];
        // Segment length is derived from the next offset; guard the last
        // segment against reading past the offsets array (mirrors CUDA).
        offsets_t segment_length;
        if (b_t == B_ * T_ - 1) {
            segment_length = static_cast<offsets_t>(permuted_indices_size_) -
                output_offsets_[b_t];
        } else {
            segment_length =
                output_offsets_[b_t + 1] - output_offsets_[b_t];
        }
        const offsets_t input_start = input_offsets_[permute_[t] * B_ + b];

        for (auto i = item.get_local_id(0); i < segment_length;
             i += item.get_local_range(0)) {
            permuted_indices_[output_start + i] = indices_[input_start + i];
            if constexpr (has_weight) {
                for (int32_t w_col = 0; w_col < weights_columns_; ++w_col) {
                    permuted_weights_
                        [(output_start + i) * weights_columns_ + w_col] =
                            weights_
                                [(input_start + i) * weights_columns_ + w_col];
                }
            }
        }
    }
}

// ============================================================================
// Host Function - XPU Implementation
// ============================================================================

std::tuple<at::Tensor, at::Tensor, std::optional<at::Tensor>>
permute_2D_sparse_preallocated_out_xpu(
    const at::Tensor& permute,
    const at::Tensor& lengths,
    const at::Tensor& indices,
    const std::optional<at::Tensor>& weights,
    const std::optional<int64_t>& permuted_lengths_sum,
    const std::optional<at::Tensor>& permuted_lengths_out,
    const std::optional<at::Tensor>& permuted_indices_out,
    const std::optional<at::Tensor>& permuted_weights_out) {

    // Device validation
    TORCH_INTERNAL_ASSERT(
        permute.device().type() == at::DeviceType::XPU,
        "permute must be on XPU device");
    TORCH_INTERNAL_ASSERT(
        lengths.device().type() == at::DeviceType::XPU,
        "lengths must be on XPU device");
    TORCH_INTERNAL_ASSERT(
        indices.device().type() == at::DeviceType::XPU,
        "indices must be on XPU device");
    TORCH_INTERNAL_ASSERT(
        !weights.has_value() ||
            weights->device().type() == at::DeviceType::XPU,
        "weights must be on XPU device");

    // Shape / dtype validation
    TORCH_CHECK(permute.dim() == 1, "permute must be 1D");
    TORCH_CHECK(lengths.dim() == 2, "lengths must be 2D");
    TORCH_CHECK(indices.dim() == 1, "indices must be 1D");
    TORCH_CHECK(permute.dtype() == at::kInt, "permute must be int32");

    const auto permute_contig = permute.contiguous();
    const auto lengths_contig = lengths.contiguous();
    const auto indices_contig = indices.contiguous();

    const int32_t T = static_cast<int32_t>(permute.numel());
    const int32_t B = static_cast<int32_t>(lengths.size(1));

    // Empty permutation or empty batch: nothing to permute over.
    if (T == 0 || B == 0) {
        return {
            lengths.clone(),
            indices.clone(),
            weights.has_value() ? std::make_optional(weights->clone())
                                : std::nullopt};
    }

    at::Tensor permuted_lengths = permuted_lengths_out.has_value()
        ? permuted_lengths_out.value()
        : at::empty({T, B}, lengths.options());

    sycl::queue& queue = c10::xpu::getCurrentXPUStream().queue();

    // ------------------------------------------------------------------------
    // Phase 1: permute lengths
    // ------------------------------------------------------------------------
    constexpr int32_t kLengthsLocal = 256;
    const int64_t lengths_total = static_cast<int64_t>(T) * B;
    const int64_t lengths_blocks =
        (lengths_total + kLengthsLocal - 1) / kLengthsLocal;
    const size_t lengths_global =
        static_cast<size_t>(lengths_blocks) * kLengthsLocal;

    AT_DISPATCH_INDEX_TYPES(
        lengths_contig.scalar_type(), "permute_2D_lengths_xpu", [&] {
            using KernelT = Permute2DLengthsKernel<index_t>;
            queue.submit([&](sycl::handler& cgh) {
                cgh.parallel_for<KernelT>(
                    sycl::nd_range<1>(
                        sycl::range<1>(lengths_global),
                        sycl::range<1>(kLengthsLocal)),
                    KernelT(
                        T,
                        B,
                        lengths_contig.data_ptr<index_t>(),
                        permute_contig.data_ptr<int32_t>(),
                        permuted_lengths.data_ptr<index_t>()));
            });
        });

    // ------------------------------------------------------------------------
    // Phase 2: convert lengths to offsets
    // ------------------------------------------------------------------------
    // Treat the [T, B] lengths as a flat sequence so the offsets line up with
    // the linear index b_t = t * B + b used inside the data kernel.
    const auto input_offsets =
        asynchronous_exclusive_cumsum_xpu(lengths_contig.flatten());
    const auto output_offsets =
        asynchronous_complete_cumsum_xpu(permuted_lengths.flatten());

    // ------------------------------------------------------------------------
    // Phase 3: determine output size
    // ------------------------------------------------------------------------
    int64_t permuted_indices_size = 0;
    if (permuted_lengths_sum.has_value()) {
        permuted_indices_size = permuted_lengths_sum.value();
    } else {
        permuted_indices_size = output_offsets[-1].item<int64_t>();
    }

    // ------------------------------------------------------------------------
    // Phase 4: allocate outputs and permute data
    // ------------------------------------------------------------------------
    at::Tensor permuted_indices = permuted_indices_out.has_value()
        ? permuted_indices_out.value()
        : at::empty(permuted_indices_size, indices.options());
    std::optional<at::Tensor> permuted_weights = std::nullopt;

    constexpr int32_t kThreadsPerSegment = 32;
    constexpr int32_t kSegmentsPerBlock = 32;
    const int64_t num_blocks =
        (static_cast<int64_t>(B) * T + kSegmentsPerBlock - 1) /
        kSegmentsPerBlock;
    const sycl::range<2> global_range{
        static_cast<size_t>(num_blocks) * kThreadsPerSegment,
        static_cast<size_t>(kSegmentsPerBlock)};
    const sycl::range<2> local_range{kThreadsPerSegment, kSegmentsPerBlock};

    if (weights.has_value()) {
        const auto weights_value = weights.value();
        const auto weights_value_contig = weights_value.contiguous();

        int32_t weights_columns = 1;
        if (weights_value.dense_dim() > 1) {
            weights_columns = static_cast<int32_t>(weights_value.size(1));
            permuted_weights = permuted_weights_out.has_value()
                ? permuted_weights_out.value()
                : at::empty(
                      {permuted_indices_size, weights_columns},
                      weights_value.options());
        } else {
            permuted_weights = permuted_weights_out.has_value()
                ? permuted_weights_out.value()
                : at::empty(
                      permuted_indices_size, weights_value.options());
        }

        AT_DISPATCH_INDEX_TYPES(
            input_offsets.scalar_type(),
            "permute_2D_data_xpu_offsets",
            [&] {
                using offsets_t = index_t;
                AT_DISPATCH_ALL_TYPES_AND2(
                    at::kHalf,
                    at::kBFloat16,
                    indices_contig.scalar_type(),
                    "permute_2D_data_xpu_indices",
                    [&] {
                        using indices_t = scalar_t;
                        AT_DISPATCH_FLOATING_TYPES_AND_HALF(
                            weights_value_contig.scalar_type(),
                            "permute_2D_data_xpu_weights",
                            [&] {
                                using weights_t = scalar_t;
                                using KernelT = Permute2DDataKernel<
                                    /*has_weight=*/true,
                                    offsets_t,
                                    indices_t,
                                    weights_t>;
                                queue.submit([&](sycl::handler& cgh) {
                                    cgh.parallel_for<KernelT>(
                                        sycl::nd_range<2>(
                                            global_range, local_range),
                                        KernelT(
                                            static_cast<int32_t>(
                                                permuted_indices_size),
                                            T,
                                            B,
                                            indices_contig
                                                .data_ptr<indices_t>(),
                                            weights_value_contig
                                                .data_ptr<weights_t>(),
                                            weights_columns,
                                            permute_contig.data_ptr<int32_t>(),
                                            input_offsets
                                                .data_ptr<offsets_t>(),
                                            output_offsets
                                                .data_ptr<offsets_t>(),
                                            permuted_indices
                                                .data_ptr<indices_t>(),
                                            permuted_weights
                                                ->data_ptr<weights_t>()));
                                });
                            });
                    });
            });
    } else {
        // No weights: instantiate the kernel with a placeholder `float`
        // weights type; the has_weight=false branch never touches weights.
        AT_DISPATCH_INDEX_TYPES(
            input_offsets.scalar_type(),
            "permute_2D_data_xpu_offsets",
            [&] {
                using offsets_t = index_t;
                AT_DISPATCH_ALL_TYPES_AND2(
                    at::kHalf,
                    at::kBFloat16,
                    indices_contig.scalar_type(),
                    "permute_2D_data_xpu_indices",
                    [&] {
                        using indices_t = scalar_t;
                        using KernelT = Permute2DDataKernel<
                            /*has_weight=*/false,
                            offsets_t,
                            indices_t,
                            float>;
                        queue.submit([&](sycl::handler& cgh) {
                            cgh.parallel_for<KernelT>(
                                sycl::nd_range<2>(global_range, local_range),
                                KernelT(
                                    static_cast<int32_t>(
                                        permuted_indices_size),
                                    T,
                                    B,
                                    indices_contig.data_ptr<indices_t>(),
                                    /*weights=*/nullptr,
                                    /*weights_columns=*/0,
                                    permute_contig.data_ptr<int32_t>(),
                                    input_offsets.data_ptr<offsets_t>(),
                                    output_offsets.data_ptr<offsets_t>(),
                                    permuted_indices.data_ptr<indices_t>(),
                                    /*permuted_weights=*/nullptr));
                        });
                    });
            });
    }

    return {permuted_lengths, permuted_indices, permuted_weights};
}

// ============================================================================
// Host Function - XPU Thin Wrapper
// ============================================================================

/**
 * Functional (allocating) entry point. Delegates to the shared implementation
 * with no pre-allocated output buffers.
 */
std::tuple<at::Tensor, at::Tensor, std::optional<at::Tensor>>
permute_2D_sparse_data_xpu(
    const at::Tensor& permute,
    const at::Tensor& lengths,
    const at::Tensor& indices,
    const std::optional<at::Tensor>& weights,
    const std::optional<int64_t>& permuted_lengths_sum) {
    return permute_2D_sparse_preallocated_out_xpu(
        permute,
        lengths,
        indices,
        weights,
        permuted_lengths_sum,
        std::nullopt,
        std::nullopt,
        std::nullopt);
}

/**
 * Register XPU implementation with PyTorch dispatch system.
 */
TORCH_LIBRARY_IMPL(fbgemm, XPU, m) {
    m.impl("permute_2D_sparse_data", &permute_2D_sparse_data_xpu);
    m.impl(
        "permute_2D_sparse_preallocated_out",
        &permute_2D_sparse_preallocated_out_xpu);
}

} // namespace fbgemm_xpu
