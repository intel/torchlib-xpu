/*
 * Copyright (c) Meta Platforms, Inc. and affiliates. All rights reserved.
 * Copyright (c) 2026 Intel Corporation. All Rights Reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */

////////////////////////////////////////////////////////////////////////////////
// SYCL PORT MAPPING TO FBGEMM CUDA SOURCE - JAGGED INDEX SELECT 2D KERNELS
////////////////////////////////////////////////////////////////////////////////
//
// This file contains SYCL kernel and host implementations for the
// jagged_index_select_2d forward operator, ported from FBGEMM CUDA.
//
// ORIGINAL CUDA SOURCE:
//   File: fbgemm_gpu/src/jagged_tensor_ops/jagged_index_select_2d_forward.cu
//
// KERNEL MAPPING:
//   JaggedIndexSelect2dKernel<scalar_t, index_t, offset_t>
//     → jagged_index_select_2d_kernel (CUDA)
//
// HOST FUNCTION MAPPING:
//   jagged_index_select_2d_forward_xpu
//     → jagged_index_select_2d_forward_cuda (CUDA)
//     CUDA File: fbgemm_gpu/src/jagged_tensor_ops/jagged_index_select_2d_forward.cu
//
////////////////////////////////////////////////////////////////////////////////

#include <algorithm>

#include <ATen/Dispatch.h>

#include "fbgemm_utils/dispatch_macros.h"
#include "jagged_index_select_2d.h"

namespace fbgemm_xpu {

namespace {

// Helper for binary search (upper_bound)
// Finds the first index i such that data[i] > target
// range: [0, n)
template <typename T>
inline int binary_search_upper_bound(const T* data, int n, T target) {
    int left = 0;
    int right = n;
    while (left < right) {
        int mid = left + (right - left) / 2;
        if (data[mid] <= target) {
            left = mid + 1;
        } else {
            right = mid;
        }
    }
    return left;
}

}  // namespace

// ============================================================================
// SYCL Kernel Functor Implementation
// ============================================================================

////////////////////////////////////////////////////////////////////////////////
// JaggedIndexSelect2dKernel::operator() - Device Kernel
////////////////////////////////////////////////////////////////////////////////
//
// CUDA SOURCE MAPPING:
//   CUDA Kernel: jagged_index_select_2d_kernel
//   CUDA File: fbgemm_gpu/src/jagged_tensor_ops/jagged_index_select_2d_forward.cu
//
// DESCRIPTION:
//   Copies sequences from the input jagged tensor into the output tensor
//   according to the indices tensor. Uses a group-strided outer loop over
//   dense output rows, a per-group binary search (broadcast from lane 0) to
//   locate the source sequence, and a work-group-strided inner loop to copy
//   columns.
//
////////////////////////////////////////////////////////////////////////////////
template <typename scalar_t, typename index_t, typename offset_t>
void JaggedIndexSelect2dKernel<scalar_t, index_t, offset_t>::operator()(
    sycl::nd_item<1> item) const {
    const int group_id = item.get_group(0);
    const int group_range = item.get_group_range(0);
    const int local_id = item.get_local_id(0);
    const int local_range = item.get_local_range(0);

    // Map groups to output rows using a grid-stride loop.
    for (offset_t dense_output_offset = group_id;
         dense_output_offset < num_dense_output_rows_;
         dense_output_offset += group_range) {

        // Binary search in the first lane, then broadcast to the work-group.
        int index_pos = 0;
        if (local_id == 0) {
            index_pos = binary_search_upper_bound(
                output_offsets_,
                static_cast<int>(num_output_rows_),
                dense_output_offset);
        }
        index_pos = sycl::group_broadcast(item.get_group(), index_pos, 0);

        // Compute source/destination row offsets.
        const offset_t prev_offset =
            (index_pos == 0) ? offset_t(0) : output_offsets_[index_pos - 1];
        const offset_t rel_index = dense_output_offset - prev_offset;

        const index_t index = indices_[index_pos];
        const offset_t input_start_offset =
            (index == 0) ? offset_t(0) : input_offsets_[index - 1];
        const offset_t input_offset = input_start_offset + rel_index;

        // Strided copy across columns.
        for (int64_t i = local_id; i < num_cols_; i += local_range) {
            output_[dense_output_offset * num_cols_ + i] =
                values_[input_offset * num_cols_ + i];
        }
    }
}

// ============================================================================
// Host Function - XPU Implementation
// ============================================================================

////////////////////////////////////////////////////////////////////////////////
// jagged_index_select_2d_forward_xpu - Host Function
////////////////////////////////////////////////////////////////////////////////
//
// CUDA SOURCE MAPPING:
//   CUDA Function: jagged_index_select_2d_forward_cuda
//   CUDA File: fbgemm_gpu/src/jagged_tensor_ops/jagged_index_select_2d_forward.cu
//
// DESCRIPTION:
//   Validates inputs, allocates the dense output tensor, and dispatches the
//   templated JaggedIndexSelect2dKernel over floating scalar types and index
//   types. Launches at most 65535 work-groups of 256 items each.
//
////////////////////////////////////////////////////////////////////////////////
at::Tensor jagged_index_select_2d_forward_xpu(
    const at::Tensor& values,
    const at::Tensor& indices,
    const at::Tensor& input_offsets,
    const at::Tensor& output_offsets,
    int64_t num_dense_output_rows) {

    TORCH_CHECK(values.is_xpu(), "values must be an XPU tensor");
    TORCH_CHECK(indices.is_xpu(), "indices must be an XPU tensor");
    TORCH_CHECK(input_offsets.is_xpu(), "input_offsets must be an XPU tensor");
    TORCH_CHECK(output_offsets.is_xpu(), "output_offsets must be an XPU tensor");

    TORCH_CHECK(values.dim() == 2, "values must be a 2D tensor");
    TORCH_CHECK(indices.dim() == 1, "indices must be a 1D tensor");
    TORCH_CHECK(input_offsets.dim() == 1, "input_offsets must be a 1D tensor");
    TORCH_CHECK(output_offsets.dim() == 1, "output_offsets must be a 1D tensor");

    const int64_t num_cols = values.size(1);
    const int64_t num_output_rows = indices.size(0);

    auto output =
        at::empty({num_dense_output_rows, num_cols}, values.options());

    if (num_dense_output_rows == 0) {
        return output;
    }

    // The kernel indexes inputs as raw row-major buffers, so materialize
    // contiguous copies for any non-contiguous input to match the CUDA
    // reference (which uses PackedTensorAccessor and honors strides).
    const auto values_contig = values.contiguous();
    const auto indices_contig = indices.contiguous();
    const auto input_offsets_contig = input_offsets.contiguous();
    const auto output_offsets_contig = output_offsets.contiguous();

    // Kernel configuration
    constexpr int64_t kWorkGroupSize = 256;
    int64_t num_groups =
        std::min(static_cast<int64_t>(65535), num_dense_output_rows);
    if (num_groups == 0) {
        num_groups = 1;
    }

    sycl::queue& queue = c10::xpu::getCurrentXPUStream().queue();

    FBGEMM_DISPATCH_ALL_TYPES(
        values.scalar_type(), "jagged_index_select_2d_forward_xpu", [&] {
            AT_DISPATCH_INDEX_TYPES(
                indices.scalar_type(),
                "jagged_index_select_2d_forward_xpu_indices",
                [&] {
                    using offset_t = int64_t;

                    queue.submit([&](sycl::handler& cgh) {
                        cgh.parallel_for<
                            JaggedIndexSelect2dKernel<scalar_t, index_t, offset_t>>(
                            sycl::nd_range<1>(
                                sycl::range<1>(num_groups * kWorkGroupSize),
                                sycl::range<1>(kWorkGroupSize)),
                            JaggedIndexSelect2dKernel<scalar_t, index_t, offset_t>(
                                output.data_ptr<scalar_t>(),
                                values_contig.data_ptr<scalar_t>(),
                                indices_contig.data_ptr<index_t>(),
                                input_offsets_contig.data_ptr<offset_t>(),
                                output_offsets_contig.data_ptr<offset_t>(),
                                num_dense_output_rows,
                                num_output_rows,
                                num_cols));
                    });
                });
        });

    return output;
}

/**
 * Register XPU implementation with PyTorch dispatch system.
 *
 * Binds the SYCL/XPU implementation to the operator schema defined in
 * ops_registry.cpp. When an XPU tensor is passed to the operator, this
 * implementation is invoked.
 */
TORCH_LIBRARY_IMPL(fbgemm, XPU, m) {
    m.impl(
        "jagged_index_select_2d_forward",
        &jagged_index_select_2d_forward_xpu);
}

}  // namespace fbgemm_xpu
