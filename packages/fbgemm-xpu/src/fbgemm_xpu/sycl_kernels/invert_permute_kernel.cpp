/*
 * Copyright (c) Meta Platforms, Inc. and affiliates. All rights reserved.
 * Copyright (c) 2026 Intel Corporation. All Rights Reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */

#include "invert_permute_kernel.h"

namespace fbgemm_xpu {

// ============================================================================
// SYCL Kernel Functor Implementations
// ============================================================================

template <typename index_t>
void InvertPermuteKernel<index_t>::operator()(const sycl::nd_item<1>& item) const {
    const int64_t global_id = item.get_global_id(0);
    const int64_t global_range = item.get_global_range(0);

    // Grid-stride loop pattern for scalability
    for (int64_t i = global_id; i < numel_; i += global_range) {
        const index_t target_idx = permute_[i];
        inversed_permute_[target_idx] = static_cast<index_t>(i);
    }
}

// ============================================================================
// Host Function - XPU Implementation
// ============================================================================
at::Tensor invert_permute_forward_xpu(const at::Tensor& permute) {
    // Input validation
    TORCH_CHECK(permute.dim() == 1,
                "invert_permute: input must be 1-dimensional, got ", permute.dim(), "D");
    TORCH_CHECK(permute.dtype() == at::kInt || permute.dtype() == at::kLong,
                "invert_permute: input must be int32 or int64, got ", permute.dtype());
    TORCH_INTERNAL_ASSERT(permute.device().type() == at::DeviceType::XPU,
                         "invert_permute_forward_xpu: input must be on XPU device");
    
    // Get input size
    const int64_t N = permute.size(0);
    
    // Handle empty tensor
    if (N == 0) {
        return at::empty_like(permute);
    }
    
    // Ensure input is contiguous for efficient memory access
    at::Tensor permute_contig = permute.contiguous();
    
    // Allocate output tensor on the same device
    at::Tensor inversed_permute = at::empty_like(permute_contig);
    
    // Get SYCL queue from current XPU stream
    sycl::queue& queue = c10::xpu::getCurrentXPUStream().queue();
    
    // Kernel configuration
    // Work-group size (local size): number of work-items per work-group
    constexpr int threads = 256;
    // Global size: total number of work-items across all work-groups
    // Rounded up to multiple of work-group size
    const int blocks = (N + threads - 1) / threads;
    const size_t global_size = blocks * threads;
    const size_t local_size = threads;
    
    // Dispatch over the index type (int32_t or int64_t), mirroring the
    // reference CUDA kernel invert_permute_kernel<index_t>.
    AT_DISPATCH_INDEX_TYPES(
        permute_contig.scalar_type(), "invert_permute_kernel", [&] {
            const index_t* permute_ptr = permute_contig.data_ptr<index_t>();
            index_t* inversed_ptr = inversed_permute.data_ptr<index_t>();

            queue.submit([&](sycl::handler& cgh) {
                cgh.parallel_for<InvertPermuteKernel<index_t>>(
                    sycl::nd_range<1>(
                        sycl::range<1>(global_size),  // Global range
                        sycl::range<1>(local_size)    // Local range
                    ),
                    InvertPermuteKernel<index_t>(N, permute_ptr, inversed_ptr)
                );
            });
        });
    
    // Note: We don't call queue.wait() here for asynchronous execution.
    // PyTorch's stream synchronization will handle proper ordering.
    
    return inversed_permute;
}

/**
 * Register XPU implementation with PyTorch dispatch system
 * 
 * This binds the SYCL/XPU implementation to the operator schema defined in ops_registry.cpp
 * When an XPU tensor is passed to the operator, this implementation will be called.
 */
TORCH_LIBRARY_IMPL(fbgemm, XPU, m) {
    m.impl("invert_permute", &invert_permute_forward_xpu);
}

} // namespace fbgemm_xpu
