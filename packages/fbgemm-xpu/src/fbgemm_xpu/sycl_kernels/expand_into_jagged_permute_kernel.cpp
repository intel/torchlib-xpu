/*
 * Copyright (c) Meta Platforms, Inc. and affiliates. All rights reserved.
 * Copyright (c) 2026 Intel Corporation. All Rights Reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */

////////////////////////////////////////////////////////////////////////////////
// SYCL PORT MAPPING TO FBGEMM CUDA SOURCE
////////////////////////////////////////////////////////////////////////////////
//
// ORIGINAL CUDA SOURCE:
//   File: fbgemm_gpu/src/sparse_ops/sparse_expand_into_jagged_permute.cu
//
// KERNEL IMPLEMENTATION:
//   ExpandIntoJaggedPermuteKernel::operator()
//     -> expand_into_jagged_permute_kernel<index_t, offsets_t> (CUDA)
//
// HOST FUNCTION:
//   expand_into_jagged_permute_xpu
//     -> expand_into_jagged_permute_cuda (CUDA)
//
////////////////////////////////////////////////////////////////////////////////

#include "expand_into_jagged_permute_kernel.h"

namespace fbgemm_xpu {

// ============================================================================
// SYCL Kernel Functor Implementation
// ============================================================================

/**
 * @brief ExpandIntoJaggedPermuteKernel operator implementation
 *
 * Mirrors the CUDA kernel expand_into_jagged_permute_kernel, with the CUDA
 * dimension order reversed because SYCL varies the *last* nd_item dimension
 * fastest while CUDA varies threadIdx.x fastest:
 *   - dimension 0 walks tables within a work-group (CUDA threadIdx.y),
 *     combined with a grid-stride loop over tables.
 *   - dimension 1 walks elements within a segment (CUDA threadIdx.x), so
 *     neighbouring work-items store to neighbouring output_permute entries.
 */
template <typename index_t, typename offsets_t>
void ExpandIntoJaggedPermuteKernel<index_t, offsets_t>::operator()(
    const sycl::nd_item<2>& item) const {
    // Grid-stride loop bounds over tables (mirrors CUDA blockIdx.x/blockDim.y).
    const int32_t t_start = static_cast<int32_t>(
        item.get_group(0) * item.get_local_range(0) + item.get_local_id(0));
    const int32_t stride = static_cast<int32_t>(
        item.get_group_range(0) * item.get_local_range(0));

    // Within-segment work-item coordinates (mirrors CUDA threadIdx.x).
    const offsets_t tid = static_cast<offsets_t>(item.get_local_id(1));
    const offsets_t threads_per_segment =
        static_cast<offsets_t>(item.get_local_range(1));

    for (int32_t t = t_start; t < input_size_; t += stride) {
        const offsets_t output_start = output_offsets_[t];
        const offsets_t segment_length =
            output_offsets_[t + 1] - output_offsets_[t];
        const offsets_t input_start = input_offsets_[permute_[t]];
        for (offsets_t i = tid; i < segment_length; i += threads_per_segment) {
            output_permute_[output_start + i] =
                static_cast<index_t>(input_start + i);
        }
    }
}

// ============================================================================
// Host Function - XPU Implementation
// ============================================================================

at::Tensor expand_into_jagged_permute_xpu(
    const at::Tensor& permute,
    const at::Tensor& input_offsets,
    const at::Tensor& output_offsets,
    int64_t output_size) {
    // Device validation
    TORCH_INTERNAL_ASSERT(
        permute.device().type() == at::DeviceType::XPU,
        "expand_into_jagged_permute_xpu: permute must be on XPU device");
    TORCH_INTERNAL_ASSERT(
        input_offsets.device().type() == at::DeviceType::XPU,
        "expand_into_jagged_permute_xpu: input_offsets must be on XPU device");
    TORCH_INTERNAL_ASSERT(
        output_offsets.device().type() == at::DeviceType::XPU,
        "expand_into_jagged_permute_xpu: output_offsets must be on XPU device");
    TORCH_CHECK(
        permute.device() == input_offsets.device() &&
            permute.device() == output_offsets.device(),
        "expand_into_jagged_permute_xpu: all input tensors must be on the "
        "same XPU device, but got permute on ",
        permute.device(),
        ", input_offsets on ",
        input_offsets.device(),
        ", and output_offsets on ",
        output_offsets.device());

    // Input validation (mirrors the CUDA TORCH_CHECKs).
    TORCH_CHECK(permute.numel() > 0);
    TORCH_CHECK(permute.numel() == input_offsets.numel() - 1);
    TORCH_CHECK(permute.numel() == output_offsets.numel() - 1);

    // Ensure contiguous for direct pointer access.
    const auto permute_contig = permute.contiguous();
    const auto input_offsets_contig = input_offsets.contiguous();
    const auto output_offsets_contig = output_offsets.contiguous();

    const int64_t permute_size = permute.numel();

    at::Tensor output_permute = at::empty({output_size}, permute.options());

    // Use the current stream for the inputs' XPU device. The process-wide
    // current device can differ in multi-XPU applications.
    sycl::queue& queue = c10::xpu::getCurrentXPUStream(
        permute.device().index()).queue();

    // Work-group layout mirrors the CUDA dim3(kWarpSize, T_blocks) launch with
    // the dimension order reversed, because SYCL varies the last nd_range
    // dimension fastest:
    //   dim 0 -> tables per work-group (kMaxThreads / kWarpSize = 32), CUDA .y
    //   dim 1 -> within-segment work-items (kWarpSize = 32), CUDA .x
    constexpr int32_t kThreadsPerSegment = 32;
    constexpr int32_t kTablesPerBlock = 32;
    const int64_t num_blocks =
        (permute_size + kTablesPerBlock - 1) / kTablesPerBlock;

    sycl::range<2> global_range{
        static_cast<size_t>(num_blocks * kTablesPerBlock),
        static_cast<size_t>(kThreadsPerSegment)};
    sycl::range<2> local_range{
        static_cast<size_t>(kTablesPerBlock),
        static_cast<size_t>(kThreadsPerSegment)};

    // Mirrors the CUDA dispatch, which binds offsets_t = index_t while keeping
    // the two template parameters distinct.
    AT_DISPATCH_INDEX_TYPES(
        permute.scalar_type(), "expand_into_jagged_permute_xpu", [&] {
            using offsets_t = index_t;
            queue.submit([&](sycl::handler& cgh) {
                cgh.parallel_for<
                    ExpandIntoJaggedPermuteKernel<index_t, offsets_t>>(
                    sycl::nd_range<2>(global_range, local_range),
                    ExpandIntoJaggedPermuteKernel<index_t, offsets_t>(
                        input_offsets_contig.data_ptr<offsets_t>(),
                        output_offsets_contig.data_ptr<offsets_t>(),
                        static_cast<int32_t>(permute_size),
                        permute_contig.data_ptr<index_t>(),
                        output_permute.data_ptr<index_t>()));
            });
        });

    return output_permute;
}

/**
 * Register XPU implementation with PyTorch dispatch system.
 */
TORCH_LIBRARY_IMPL(fbgemm, XPU, m) {
    m.impl("expand_into_jagged_permute", &expand_into_jagged_permute_xpu);
}

} // namespace fbgemm_xpu
