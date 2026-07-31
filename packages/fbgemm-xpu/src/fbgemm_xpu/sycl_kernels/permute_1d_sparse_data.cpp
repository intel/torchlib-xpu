/*
 * Copyright (c) Meta Platforms, Inc. and affiliates. All rights reserved.
 * Copyright (c) 2026 Intel Corporation. All Rights Reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */

/*
 * SYCL/XPU Implementation of permute_1D_sparse_data operator
 *
 * This operator permutes sparse data represented in jagged/1D format.
 * It reorders (permutes) segments of data according to a permutation array.
 *
 * The operation is commonly used in:
 * - Deep Learning Recommendation Models (DLRMs) for reordering embedding features
 * - Sparse tensor operations in graph neural networks
 * - Distributed training for all-to-all communication preparation
 *
 * Input format (jagged/CSR-like):
 *   lengths: [L0, L1, L2, ...]  - length of each segment
 *   values:  [seg0 data][seg1 data][seg2 data]...  - concatenated values
 *   permute: [P0, P1, P2, ...]  - new segment order
 *
 * Output:
 *   permuted_lengths[j] = lengths[permute[j]]
 *   permuted_values contains data from segment permute[j] at position j
 */

////////////////////////////////////////////////////////////////////////////////
// SYCL PORT MAPPING TO FBGEMM CUDA SOURCE
////////////////////////////////////////////////////////////////////////////////
//
// This file contains SYCL implementations of FBGEMM sparse data permutation
// kernels and host functions.
//
// ORIGINAL CUDA SOURCE:
//   File: fbgemm_gpu/src/sparse_ops/sparse_permute_1d.cu
//
// KERNEL IMPLEMENTATIONS:
//   Permute1DLengthsKernel::operator()
//     → permute_1D_lengths_kernel (CUDA)
//
//   Permute1DDataKernel::operator()
//     → permute_1D_data_kernel<has_weight, ...> (CUDA)
//
// HOST FUNCTION:
//   permute_1D_sparse_data_xpu
//     → permute_1D_sparse_data_cuda (CUDA)
//
// HELPER FUNCTIONS:
//   exclusive_cumsum_xpu
//     → asynchronous_exclusive_cumsum_gpu (CUDA)
//
//   complete_cumsum_xpu
//     → asynchronous_complete_cumsum_gpu (CUDA)
//
////////////////////////////////////////////////////////////////////////////////

#include "permute_1d_sparse_data.h"

namespace fbgemm_xpu {

// ============================================================================
// SYCL Kernel Functors - Operator Implementations
// ============================================================================

/**
 * @brief Permute1DLengthsKernel operator implementation
 */
template <typename index_t, typename permute_t>
void Permute1DLengthsKernel<index_t, permute_t>::operator()(
    const sycl::nd_item<1>& item) const {
    const int64_t i = item.get_global_id(0);
    if (i < permuted_lengths_size_) {
        permuted_lengths_[i] = lengths_[permute_[i]];
    }
}

/**
 * @brief Permute1DDataKernel operator implementation
 *
 * Copies indices for each segment; when `has_weight` is true it also copies
 * the associated weights. The weight-copy branch is a compile-time `if
 * constexpr`, so the no-weights instantiation pays no runtime cost.
 */
template <
    bool has_weight,
    typename offsets_t,
    typename indices_t,
    typename weights_t>
void Permute1DDataKernel<has_weight, offsets_t, indices_t, weights_t>::operator()(
    const sycl::nd_item<2>& item) const {
    // Get segment ID and thread ID within segment
    const int32_t segment_base = item.get_group(0) * item.get_local_range(1);
    const int32_t segment_local = item.get_local_id(1);
    const int32_t segment_id = segment_base + segment_local;
    const int32_t tid = item.get_local_id(0);
    const int32_t threads_per_segment = item.get_local_range(0);

    if (segment_id >= permuted_lengths_size_) {
        return;
    }

    // Calculate segment boundaries
    const offsets_t output_start = output_offsets_[segment_id];
    const offsets_t output_end = (segment_id == permuted_lengths_size_ - 1)
        ? static_cast<offsets_t>(permuted_indices_size_)
        : output_offsets_[segment_id + 1];
    const int32_t segment_length = static_cast<int32_t>(output_end - output_start);
    const offsets_t input_start = input_offsets_[permute_[segment_id]];

    // Copy data using strided access pattern
    for (int32_t i = tid; i < segment_length; i += threads_per_segment) {
        permuted_indices_[output_start + i] = indices_[input_start + i];
        if constexpr (has_weight) {
            // Copy all columns for this row (supports both 1D and 2D weights)
            const int64_t input_row = input_start + i;
            const int64_t output_row = output_start + i;
            for (int64_t col = 0; col < weight_columns_; ++col) {
                permuted_weights_[output_row * weight_columns_ + col] = 
                    weights_[input_row * weight_columns_ + col];
            }
        }
    }
}

// ============================================================================
// Helper Functions
// ============================================================================

////////////////////////////////////////////////////////////////////////////////
// exclusive_cumsum_xpu - Helper Function
////////////////////////////////////////////////////////////////////////////////
//
// CUDA SOURCE MAPPING:
//   CUDA Function: asynchronous_exclusive_cumsum_gpu
//   CUDA File: fbgemm_gpu/src/sparse_ops/sparse_ops_gpu.cu
//
// DESCRIPTION:
//   Computes exclusive prefix sum (cumulative sum with 0 prepended).
//   Used to convert lengths array to offsets array for jagged tensor access.
//
//   Example: input = [3, 2, 4] → output = [0, 3, 5, 9]
//
////////////////////////////////////////////////////////////////////////////////

/**
 * @brief Compute exclusive cumsum (prefix sum) for offsets
 *
 * output[0] = 0
 * output[i] = sum(input[0:i]) for i > 0
 */
template <typename T>
at::Tensor exclusive_cumsum_xpu(const at::Tensor& input) {
    auto output = at::empty({input.numel() + 1}, input.options().dtype(at::kLong));
    output[0] = 0;
    if (input.numel() > 0) {
        auto cumsum = input.cumsum(0, at::kLong);
        output.slice(0, 1).copy_(cumsum);
    }
    return output;
}

////////////////////////////////////////////////////////////////////////////////
// complete_cumsum_xpu - Helper Function
////////////////////////////////////////////////////////////////////////////////
//
// CUDA SOURCE MAPPING:
//   CUDA Function: asynchronous_complete_cumsum_gpu
//   CUDA File: fbgemm_gpu/src/sparse_ops/sparse_ops_gpu.cu
//
// DESCRIPTION:
//   Computes complete cumulative sum (inclusive prefix sum with 0 prepended).
//   Used to convert permuted lengths to output offsets.
//
//   Identical to exclusive_cumsum_xpu in current implementation.
//   Example: input = [4, 3, 2] → output = [0, 4, 7, 9]
//
////////////////////////////////////////////////////////////////////////////////

/**
 * @brief Compute complete cumsum (inclusive + final element)
 *
 * output[0] = 0
 * output[i] = sum(input[0:i]) for i > 0
 * output[n] = sum(all input)
 */
template <typename T>
at::Tensor complete_cumsum_xpu(const at::Tensor& input) {
    auto output = at::empty({input.numel() + 1}, input.options().dtype(at::kLong));
    output[0] = 0;
    if (input.numel() > 0) {
        auto cumsum = input.cumsum(0, at::kLong);
        output.slice(0, 1).copy_(cumsum);
    }
    return output;
}

// ============================================================================
// Host Function - XPU Implementation
// ============================================================================

////////////////////////////////////////////////////////////////////////////////
// permute_1D_sparse_data_xpu - Host Function
////////////////////////////////////////////////////////////////////////////////
//
// CUDA SOURCE MAPPING:
//   CUDA Function: permute_1D_sparse_data_cuda
//   CUDA File: fbgemm_gpu/src/sparse_ops/sparse_permute_1d.cu
//
// DESCRIPTION:
//   Main host function that orchestrates permutation of sparse jagged data.
//
////////////////////////////////////////////////////////////////////////////////

/**
 * @brief XPU implementation of permute_1D_sparse_data
 *
 * Permutes sparse data represented in jagged format according to permutation indices.
 *
 * @param permute Permutation indices tensor [P] - int32
 * @param lengths Segment lengths tensor [L] - int32/int64
 * @param indices Concatenated values tensor [V] - any type
 * @param weights Optional weights tensor [V] - float/double
 * @param permuted_lengths_sum Optional precomputed sum of permuted lengths
 * @return Tuple of (permuted_lengths, permuted_indices, permuted_weights)
 */
std::tuple<at::Tensor, at::Tensor, std::optional<at::Tensor>>
permute_1D_sparse_data_xpu(
    const at::Tensor& permute,
    const at::Tensor& lengths,
    const at::Tensor& indices,
    const std::optional<at::Tensor>& weights,
    const std::optional<int64_t>& permuted_lengths_sum) {

    // Device validation
    TORCH_INTERNAL_ASSERT(permute.device().type() == at::DeviceType::XPU,
                         "permute must be on XPU device");
    TORCH_INTERNAL_ASSERT(lengths.device().type() == at::DeviceType::XPU,
                         "lengths must be on XPU device");
    TORCH_INTERNAL_ASSERT(indices.device().type() == at::DeviceType::XPU,
                         "indices must be on XPU device");

    // Input validation
    TORCH_CHECK(permute.dim() == 1, "permute must be 1D");
    TORCH_CHECK(lengths.dim() == 1, "lengths must be 1D");
    TORCH_CHECK(indices.dim() == 1, "indices must be 1D");
    TORCH_CHECK(permute.dtype() == at::kInt, "permute must be int32");

    // Ensure contiguous
    const auto permute_contig = permute.contiguous();
    const auto lengths_contig = lengths.contiguous();
    const auto indices_contig = indices.contiguous();

    const int64_t permuted_lengths_size = permute.numel();
    const int64_t lengths_size = lengths.numel();

    // Handle empty input
    if (permuted_lengths_size == 0) {
        // Empty permutation returns empty outputs
        return {
            at::empty({0}, lengths.options()),
            at::empty({0}, indices.options()),
            weights.has_value() ? std::make_optional(at::empty({0}, weights->options())) : std::nullopt
        };
    }

    if (lengths_size == 0) {
        // Empty lengths but non-empty permute - return zeros for lengths and empty indices
        // Warning: if permute contains valid indices they would be out of bounds,
        // but typically 0 lengths implies 0 segments, so permute should be empty too.
        // If we really want to support "selecting from 0 segments" it implies permute indices are invalid
        // unless we assume they map to implicit 0 length segments?
        // Following CPU logic:
        return {
            at::zeros({permuted_lengths_size}, lengths.options()),
            at::empty({0}, indices.options()),
            weights.has_value() ? std::make_optional(at::empty({0}, weights->options())) : std::nullopt
        };
    }

    // Get SYCL queue
    sycl::queue& queue = c10::xpu::getCurrentXPUStream().queue();

    // Phase 1: Allocate and compute permuted_lengths
    at::Tensor permuted_lengths = at::empty({permuted_lengths_size}, lengths.options());

    // Launch lengths permutation kernel
    constexpr int kLocalSize = 256;
    const int64_t global_size = ((permuted_lengths_size + kLocalSize - 1) / kLocalSize) * kLocalSize;

    AT_DISPATCH_INDEX_TYPES(
    permute.scalar_type(), "permute_1D_lengths_permute_type", [&] {
    using permute_t = index_t;
        AT_DISPATCH_INDEX_TYPES(
            lengths.scalar_type(), "permute_1D_lengths_xpu", [&] {
                queue.submit([&](sycl::handler& cgh) {
                    cgh.parallel_for<Permute1DLengthsKernel<index_t, permute_t>>(
                        sycl::nd_range<1>(
                            sycl::range<1>(global_size),
                            sycl::range<1>(kLocalSize)
                        ),
                        Permute1DLengthsKernel<index_t, permute_t>(
                            permuted_lengths_size,
                            lengths_contig.data_ptr<index_t>(),
                            permute_contig.data_ptr<permute_t>(),
                            permuted_lengths.data_ptr<index_t>()
                        )
                    );
                });
            }
        );
    });

    // Phase 2: Compute input and output offsets
    at::Tensor input_offsets = exclusive_cumsum_xpu<int64_t>(lengths_contig);
    at::Tensor output_offsets = complete_cumsum_xpu<int64_t>(permuted_lengths);

    // Phase 3: Determine output size
    int64_t permuted_indices_size = 0;
    if (permuted_lengths_sum.has_value()) {
        permuted_indices_size = permuted_lengths_sum.value();
    } else {
        // Need to sync to get the value from device
        permuted_indices_size = output_offsets[permuted_lengths_size].item<int64_t>();
    }

    // Phase 4: Allocate output tensors
    at::Tensor permuted_indices = at::empty(permuted_indices_size, indices.options());
    std::optional<at::Tensor> permuted_weights = std::nullopt;

    // Phase 5: Launch data permutation kernel
    constexpr int32_t kThreadsPerSegment = 64;
    constexpr int32_t kSegmentsPerBlock = 16;

    const int64_t num_blocks = (permuted_lengths_size + kSegmentsPerBlock - 1) / kSegmentsPerBlock;
    sycl::range<2> global_range{static_cast<size_t>(num_blocks * kThreadsPerSegment),
                                 static_cast<size_t>(kSegmentsPerBlock)};
    sycl::range<2> local_range{kThreadsPerSegment, kSegmentsPerBlock};

    if (weights.has_value()) {
        // With weights - preserve dimensionality (1D or 2D)
        const auto weights_contig = weights->contiguous();
        const int64_t weight_columns = (weights->dim() == 2) ? weights->size(1) : 1;
        
        // Create output tensor with same shape as input weights
        if (weights->dim() == 2) {
            permuted_weights = at::empty({permuted_indices_size, weight_columns}, weights->options());
        } else {
            permuted_weights = at::empty(permuted_indices_size, weights->options());
        }

        AT_DISPATCH_INDEX_TYPES(
            input_offsets.scalar_type(), "permute_1D_data_xpu_offsets", [&] {
                using offsets_t = index_t;
                AT_DISPATCH_ALL_TYPES_AND2(
                    at::kHalf, at::kBFloat16,
                    indices.scalar_type(), "permute_1D_data_xpu_indices", [&] {
                        using indices_t = scalar_t;
                        AT_DISPATCH_FLOATING_TYPES_AND_HALF(
                            weights->scalar_type(), "permute_1D_data_xpu_weights", [&] {
                                using weights_t = scalar_t;
                                using KernelT = Permute1DDataKernel<
                                    /*has_weight=*/true, offsets_t, indices_t, weights_t>;
                                queue.submit([&](sycl::handler& cgh) {
                                    cgh.parallel_for<KernelT>(
                                        sycl::nd_range<2>(global_range, local_range),
                                        KernelT(
                                            permuted_indices_size,
                                            permuted_lengths_size,
                                            indices_contig.data_ptr<indices_t>(),
                                            weights_contig.data_ptr<weights_t>(),
                                            permute_contig.data_ptr<int32_t>(),
                                            input_offsets.data_ptr<offsets_t>(),
                                            output_offsets.data_ptr<offsets_t>(),
                                            permuted_indices.data_ptr<indices_t>(),
                                            permuted_weights->data_ptr<weights_t>(),
                                            weight_columns
                                        )
                                    );
                                });
                            }
                        );
                    }
                );
            }
        );
    } else {
        // Without weights -- weights_t is a placeholder (float) since
        // has_weight=false disables the weight-copy branch at compile time.
        AT_DISPATCH_INDEX_TYPES(
            input_offsets.scalar_type(), "permute_1D_data_xpu_offsets", [&] {
                using offsets_t = index_t;
                AT_DISPATCH_ALL_TYPES_AND2(
                    at::kHalf, at::kBFloat16,
                    indices.scalar_type(), "permute_1D_data_xpu_indices", [&] {
                        using indices_t = scalar_t;
                        using KernelT = Permute1DDataKernel<
                            /*has_weight=*/false, offsets_t, indices_t, float>;
                        queue.submit([&](sycl::handler& cgh) {
                            cgh.parallel_for<KernelT>(
                                sycl::nd_range<2>(global_range, local_range),
                                KernelT(
                                    permuted_indices_size,
                                    permuted_lengths_size,
                                    indices_contig.data_ptr<indices_t>(),
                                    /*weights=*/nullptr,
                                    permute_contig.data_ptr<int32_t>(),
                                    input_offsets.data_ptr<offsets_t>(),
                                    output_offsets.data_ptr<offsets_t>(),
                                    permuted_indices.data_ptr<indices_t>(),
                                    /*permuted_weights=*/nullptr
                                )
                            );
                        });
                    }
                );
            }
        );
    }

    return {permuted_lengths, permuted_indices, permuted_weights};
}

/**
 * Register XPU implementation with PyTorch dispatch system
 */
TORCH_LIBRARY_IMPL(fbgemm, XPU, m) {
    m.impl("permute_1D_sparse_data", &permute_1D_sparse_data_xpu);
}

} // namespace fbgemm_xpu
