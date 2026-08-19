/*
 * Copyright (c) Meta Platforms, Inc. and affiliates. All rights reserved.
 * Copyright (c) 2026 Intel Corporation. All Rights Reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */

////////////////////////////////////////////////////////////////////////////////
// SYCL PORT MAPPING TO FBGEMM CUDA SOURCE - EXPAND INTO JAGGED PERMUTE OPERATOR
////////////////////////////////////////////////////////////////////////////////
//
// This file contains SYCL ports of the FBGEMM expand_into_jagged_permute
// operator.
//
// ORIGINAL CUDA SOURCE:
//   File: fbgemm_gpu/src/sparse_ops/sparse_expand_into_jagged_permute.cu
//
// KERNEL MAPPING:
//   ExpandIntoJaggedPermuteKernel<index_t> (SYCL)
//     -> expand_into_jagged_permute_kernel<index_t, offsets_t> (CUDA)
//
// HOST FUNCTION MAPPING:
//   expand_into_jagged_permute_xpu (SYCL)
//     -> expand_into_jagged_permute_cuda (CUDA)
//
// DESCRIPTION:
//   Expands a per-table (segment) permutation into a flat, element-wise index
//   permutation over jagged data. For each table t the kernel copies the
//   contiguous run of indices belonging to segment permute[t] from the input
//   layout to the output layout:
//
//       output_permute[output_offsets[t] + i] = input_offsets[permute[t]] + i
//
//   for i in [0, segment_length). The CUDA kernel uses a 2D thread block
//   (threadIdx.y over tables, threadIdx.x within a segment) combined with a
//   grid-stride loop over tables; this SYCL port mirrors that layout 1:1 using
//   a 2D nd_range (local_id(1) over tables, local_id(0) within a segment).
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
// ExpandIntoJaggedPermuteKernel - Device Kernel
////////////////////////////////////////////////////////////////////////////////
//
// CUDA SOURCE MAPPING:
//   CUDA Kernel: expand_into_jagged_permute_kernel<index_t, offsets_t>
//   CUDA File: fbgemm_gpu/src/sparse_ops/sparse_expand_into_jagged_permute.cu
//
// DESCRIPTION:
//   Templated kernel that expands a segment-level permutation into a flat
//   index permutation. The reference CUDA kernel is templated on both index_t
//   (permute / output_permute) and offsets_t (offsets); the CUDA dispatch sets
//   offsets_t = index_t, so this port uses a single index type.
//
//   Thread mapping (mirrors the CUDA dim3(kWarpSize, T_blocks) layout):
//     - dimension 1 (local_id(1)) indexes tables within a work-group
//     - dimension 0 (local_id(0)) indexes elements within a segment
//   A grid-stride loop over tables keeps correctness for any input_size,
//   matching the CUDA grid-stride loop over t.
//
////////////////////////////////////////////////////////////////////////////////
template <typename index_t>
class ExpandIntoJaggedPermuteKernel {
public:
    ExpandIntoJaggedPermuteKernel(
        const index_t* input_offsets,
        const index_t* output_offsets,
        int32_t input_size,
        const index_t* permute,
        index_t* output_permute)
        : input_offsets_(input_offsets),
          output_offsets_(output_offsets),
          input_size_(input_size),
          permute_(permute),
          output_permute_(output_permute) {}

    void operator()(const sycl::nd_item<2>& item) const;

private:
    const index_t* input_offsets_;
    const index_t* output_offsets_;
    int32_t input_size_;
    const index_t* permute_;
    index_t* output_permute_;
};

// ============================================================================
// Host Function Declaration
// ============================================================================

////////////////////////////////////////////////////////////////////////////////
// expand_into_jagged_permute_xpu - Host Function
////////////////////////////////////////////////////////////////////////////////
//
// CUDA SOURCE MAPPING:
//   CUDA Function: expand_into_jagged_permute_cuda
//   CUDA File: fbgemm_gpu/src/sparse_ops/sparse_expand_into_jagged_permute.cu
//
// DESCRIPTION:
//   XPU implementation of expand_into_jagged_permute.
//
// @param permute        Segment permutation tensor [T] (int32 or int64, XPU)
// @param input_offsets  Input segment offsets [T + 1] (same dtype as permute)
// @param output_offsets Output segment offsets [T + 1] (same dtype as permute)
// @param output_size    Total number of expanded elements (output length)
// @return at::Tensor     Flat output permutation tensor [output_size]
//
////////////////////////////////////////////////////////////////////////////////
at::Tensor expand_into_jagged_permute_xpu(
    const at::Tensor& permute,
    const at::Tensor& input_offsets,
    const at::Tensor& output_offsets,
    int64_t output_size);

} // namespace fbgemm_xpu
