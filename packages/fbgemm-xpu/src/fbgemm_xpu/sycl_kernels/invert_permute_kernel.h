/*
 * Copyright (c) Meta Platforms, Inc. and affiliates. All rights reserved.
 * Copyright (c) 2026 Intel Corporation. All Rights Reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */

////////////////////////////////////////////////////////////////////////////////
// SYCL PORT MAPPING TO FBGEMM CUDA SOURCE - INVERT PERMUTE OPERATOR
////////////////////////////////////////////////////////////////////////////////
//
// This file contains SYCL ports of FBGEMM invert_permute operator.
//
// ORIGINAL CUDA SOURCE:
//   File: fbgemm_gpu/src/sparse_ops/sparse_invert_permute.cu
//
// KERNEL MAPPING:
//   InvertPermuteKernelInt32 (SYCL)
//     → invert_permute_kernel<int32_t> (CUDA)
//   InvertPermuteKernelInt64 (SYCL)
//     → invert_permute_kernel<int64_t> (CUDA)
//
// HOST FUNCTION MAPPING:
//   invert_permute_forward_xpu (SYCL)
//     → invert_permute_cuda (CUDA)
//     CUDA File: fbgemm_gpu/src/sparse_ops/sparse_invert_permute.cu
//
// DESCRIPTION:
//   Computes the inverse of a permutation tensor on Intel XPU devices.
//   If permute[i] = j, then inversed_permute[j] = i.
//   The SYCL kernel uses a grid-stride loop pattern for scalability,
//   allowing efficient handling of permutations of any size.
//
////////////////////////////////////////////////////////////////////////////////

#pragma once

#include <cstdint>

#include <sycl/sycl.hpp>

#include <c10/xpu/XPUStream.h>

#include <ATen/ATen.h>
#include <torch/library.h>

namespace fbgemm_xpu {

// ============================================================================
// SYCL Kernel Functors
// ============================================================================

////////////////////////////////////////////////////////////////////////////////
// InvertPermuteKernelInt32 - Device Kernel
////////////////////////////////////////////////////////////////////////////////
//
// CUDA SOURCE MAPPING:
//   CUDA Kernel: invert_permute_kernel<int32_t>
//   CUDA File: fbgemm_gpu/src/sparse_ops/sparse_invert_permute.cu
//
// DESCRIPTION:
//   Main kernel for inverting a permutation tensor (int32 version).
//   Implements the core logic: inversed_permute[permute[i]] = i
//   using a grid-stride loop pattern for scalability.
//
//   The grid-stride loop allows each work-item to process multiple elements,
//   ensuring correct behavior regardless of the relationship between N
//   (number of elements) and the number of work-items launched.
//
////////////////////////////////////////////////////////////////////////////////
class InvertPermuteKernelInt32 {
public:
    InvertPermuteKernelInt32(
        int64_t numel,
        const int32_t* permute,
        int32_t* inversed_permute)
        : numel_(numel),
          permute_(permute),
          inversed_permute_(inversed_permute) {}
    
    void operator()(const sycl::nd_item<1>& item) const;

private:
    int64_t numel_;
    const int32_t* permute_;
    int32_t* inversed_permute_;
};

////////////////////////////////////////////////////////////////////////////////
// InvertPermuteKernelInt64 - Device Kernel
////////////////////////////////////////////////////////////////////////////////
//
// CUDA SOURCE MAPPING:
//   CUDA Kernel: invert_permute_kernel<int64_t>
//   CUDA File: fbgemm_gpu/src/sparse_ops/sparse_invert_permute.cu
//
// DESCRIPTION:
//   Main kernel for inverting a permutation tensor (int64 version).
//   Same logic as Int32 version but for 64-bit integers.
//   Uses grid-stride loop pattern for scalability.
//
////////////////////////////////////////////////////////////////////////////////
class InvertPermuteKernelInt64 {
public:
    InvertPermuteKernelInt64(
        int64_t numel,
        const int64_t* permute,
        int64_t* inversed_permute)
        : numel_(numel),
          permute_(permute),
          inversed_permute_(inversed_permute) {}
    
    void operator()(const sycl::nd_item<1>& item) const;

private:
    int64_t numel_;
    const int64_t* permute_;
    int64_t* inversed_permute_;
};

// ============================================================================
// Host Function Declaration
// ============================================================================

////////////////////////////////////////////////////////////////////////////////
// invert_permute_forward_xpu - Host Function
////////////////////////////////////////////////////////////////////////////////
//
// CUDA SOURCE MAPPING:
//   CUDA Function: invert_permute_cuda
//   CUDA File: fbgemm_gpu/src/sparse_ops/sparse_invert_permute.cu
//
// DESCRIPTION:
//   XPU implementation of invert_permute forward pass.
//   Computes the inverse of a permutation tensor using SYCL on Intel GPUs.
//
//   Input validation:
//   - Ensures input is 1D tensor
//   - Verifies dtype is int32 or int64
//   - Checks tensor is on XPU device
//
//   Performance:
//   - Uses grid-stride loop for scalability
//   - Work-group size: 256 threads
//   - Asynchronous execution via PyTorch XPU stream
//
// @param permute Input permutation tensor (1D, int32 or int64, on XPU device)
// @return at::Tensor Output inverse permutation tensor (same shape, dtype, device)
//
////////////////////////////////////////////////////////////////////////////////
at::Tensor invert_permute_forward_xpu(const at::Tensor& permute);

} // namespace fbgemm_xpu
