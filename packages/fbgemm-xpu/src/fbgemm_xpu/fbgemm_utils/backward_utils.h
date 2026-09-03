/*
 * Copyright (c) Meta Platforms, Inc. and affiliates. All rights reserved.
 * Copyright (c) 2026 Intel Corporation. All Rights Reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */


////////////////////////////////////////////////////////////////////////////////
// SYCL PORT MAPPING TO FBGEMM CUDA SOURCE - BACKWARD UTILITIES
////////////////////////////////////////////////////////////////////////////////
//
// This file contains SYCL ports of FBGEMM backward pass utility functions.
//
// ORIGINAL CUDA SOURCES:
//   Files: Multiple template files in training/backward/
//   Paths:
//    - FBGEMM/fbgemm_gpu/codegen/training/backward/
//    - FBGEMM/fbgemm_gpu/include/fbgemm_gpu
//   Key Files:
//     - embedding_backward_split_template.cu
//     - embedding_backward_template_helpers.cuh
//
// UTILITY MAPPING:
//   transpose_embedding_input
//     → transpose_embedding_input (CUDA)
//     Purpose: Convert CSR to CSC representation via sorting
//
//   LinearizeIndexKernel
//     → linearize_index_kernel (CUDA)
//     Purpose: Convert (table_id, index) to flat linear index
//
//   SplitEmbeddingBackwardCodegenFindLongSegments
//     → split_embedding_backward_codegen_find_long_segments (CUDA)
//     Purpose: Identify segments requiring block-level parallelism
//
//   compute_grad_sum_unweighted_nobag
//     → Gradient accumulation helper (inline in CUDA kernels)
//     Purpose: Accumulate gradients with warp-level cooperation
//
////////////////////////////////////////////////////////////////////////////////

#pragma once

#include <ATen/ATen.h>
#include <ATen/Operators.h>
#include <ATen/core/TensorAccessor.h>
#include <ATen/native/StridedRandomAccessor.h>
#include <c10/xpu/XPUStream.h>

#include <tuple>
#include <sycl/sycl.hpp>

#include "../sparse_async_cumsum.h"
#include "fbgemm_utils/utils.h"
#include "fbgemm_utils/vec4.h"

using Tensor = at::Tensor;
using at::native::RestrictPtrTraits;

namespace fbgemm_xpu {

// Pass 1: Mark run starts
template <typename index_t>
class MarkRunStartsKernel {
 public:
    MarkRunStartsKernel(const index_t* sorted_input, int32_t* run_starts,
                        int64_t total_elements)
        : sorted_input_(sorted_input),
          run_starts_(run_starts),
          total_elements_(total_elements) {}

    void operator()(const sycl::nd_item<1>& item) const;

 private:
    const index_t* sorted_input_;
    int32_t* run_starts_;
    int64_t total_elements_;
};

// Pass 2: Compact runs using prefix sum
template <typename index_t>
class CompactRunsKernel {
 public:
    CompactRunsKernel(const index_t* sorted_input, const int32_t* run_starts,
                      const int32_t* run_positions,  // prefix sum
                      index_t* unique_output, int32_t* run_lengths,
                      int64_t total_elements)
        : sorted_input_(sorted_input),
          run_starts_(run_starts),
          run_positions_(run_positions),
          unique_output_(unique_output),
          run_lengths_(run_lengths),
          total_elements_(total_elements) {}

    void operator()(const sycl::nd_item<1>& item) const;

 private:
    const index_t* sorted_input_;
    const int32_t* run_starts_;
    const int32_t* run_positions_;
    index_t* unique_output_;
    int32_t* run_lengths_;
    int64_t total_elements_;
};

////////////////////////////////////////////////////////////////////////////////
// SplitEmbeddingBackwardFindLongSegments - Segment Classification
////////////////////////////////////////////////////////////////////////////////
//
// CUDA SOURCE MAPPING:
//   CUDA Kernel: split_embedding_backward_codegen_find_long_segments
//   Template File: embedding_backward_split_grad_template.cu
//   CUDA Path: /FBGEMM/fbgemm_gpu/_skbuild/linux-x86_64-3.10/cmake-build
//   Generated Instance: gen_embedding_backward_split_grad_index_select.cu
//
// DESCRIPTION:
//   Identifies segments (run-length encoded embeddings) that require
//   block-level parallelism vs warp-level parallelism.
//
////////////////////////////////////////////////////////////////////////////////
class SplitEmbeddingBackwardFindLongSegments {
 public:
    SplitEmbeddingBackwardFindLongSegments(
        const at::PackedTensorAccessor32<int32_t, 1, RestrictPtrTraits>
            sorted_linear_indices_num_runs,
        const at::PackedTensorAccessor32<int32_t, 1, RestrictPtrTraits>
            sorted_linear_indices_run_lengths,
        at::PackedTensorAccessor32<int32_t, 1, RestrictPtrTraits> long_run_ids,
        at::PackedTensorAccessor32<int32_t, 1, RestrictPtrTraits>
            num_long_run_ids,
        at::PackedTensorAccessor32<int32_t, 1, RestrictPtrTraits>
            long_run_id_to_really_long_run_ids,
        at::PackedTensorAccessor32<int32_t, 1, RestrictPtrTraits>
            num_really_long_run_ids,
        at::PackedTensorAccessor32<int32_t, 1, RestrictPtrTraits>
            grad_accum_counter,
        const int32_t max_segment_length_per_warp,
        const int32_t max_segment_length_per_cta,
        const bool use_deterministic_algorithms)
        : sorted_linear_indices_num_runs_(sorted_linear_indices_num_runs),
          sorted_linear_indices_run_lengths_(sorted_linear_indices_run_lengths),
          long_run_ids_(long_run_ids),
          num_long_run_ids_(num_long_run_ids),
          long_run_id_to_really_long_run_ids_(
              long_run_id_to_really_long_run_ids),
          num_really_long_run_ids_(num_really_long_run_ids),
          grad_accum_counter_(grad_accum_counter),
          max_segment_length_per_warp_(max_segment_length_per_warp),
          max_segment_length_per_cta_(max_segment_length_per_cta),
          use_deterministic_algorithms_(use_deterministic_algorithms) {}

    void SYCL_EXTERNAL operator()(const sycl::nd_item<1>& item) const;

 private:
    const at::PackedTensorAccessor32<int32_t, 1, RestrictPtrTraits>
        sorted_linear_indices_num_runs_;
    const at::PackedTensorAccessor32<int32_t, 1, RestrictPtrTraits>
        sorted_linear_indices_run_lengths_;
    mutable at::PackedTensorAccessor32<int32_t, 1, RestrictPtrTraits>
        long_run_ids_;
    mutable at::PackedTensorAccessor32<int32_t, 1, RestrictPtrTraits>
        num_long_run_ids_;
    mutable at::PackedTensorAccessor32<int32_t, 1, RestrictPtrTraits>
        long_run_id_to_really_long_run_ids_;
    mutable at::PackedTensorAccessor32<int32_t, 1, RestrictPtrTraits>
        num_really_long_run_ids_;
    mutable at::PackedTensorAccessor32<int32_t, 1, RestrictPtrTraits>
        grad_accum_counter_;
    int32_t max_segment_length_per_warp_;
    int32_t max_segment_length_per_cta_;
    bool use_deterministic_algorithms_;
};

////////////////////////////////////////////////////////////////////////////////
// LinearizeIndexKernel - Index Linearization
////////////////////////////////////////////////////////////////////////////////
//
// CUDA SOURCE MAPPING:
//   CUDA Kernel: linearize_index_kernel
//   CUDA File: transpose_embedding_input.cu
//   CUDA Path: FBGEMM/fbgemm_gpu/src/split_embeddings_utils/
//
// DESCRIPTION:
//   Converts (table_id, local_index) pairs to flat linear indices.
//   Uses hash_size_cumsum as offsets for each table.
//   Supports both regular and VBE (Variable Batch Embedding) modes.
//
////////////////////////////////////////////////////////////////////////////////
template <typename index_t, typename info_acc_t, bool nobag, bool vbe>
class LinearizeIndexKernel {
 public:
    LinearizeIndexKernel(
        const at::PackedTensorAccessor32<int64_t, 1, RestrictPtrTraits>
            hash_size_cumsum,
        const at::PackedTensorAccessor32<index_t, 1, RestrictPtrTraits> indices,
        const at::PackedTensorAccessor32<index_t, 1, RestrictPtrTraits> offsets,
        at::PackedTensorAccessor32<info_acc_t, 1, RestrictPtrTraits> infos,
        at::PackedTensorAccessor32<index_t, 1, RestrictPtrTraits>
            linear_indices,
        const int32_t info_B_num_bits, const uint32_t info_B_mask,
        const uint32_t max_T, const uint32_t max_B,
        // Use a raw pointer to avoid creating dummy PackedTensorAccessor
        const uint32_t* const __restrict__ vbe_b_t_map, FixedDivisor fd)
        : hash_size_cumsum_(hash_size_cumsum),
          indices_(indices),
          offsets_(offsets),
          infos_(infos),
          linear_indices_(linear_indices),
          info_B_num_bits_(info_B_num_bits),
          info_B_mask_(info_B_mask),
          max_T_(max_T),
          max_B_(max_B),
          vbe_b_t_map_(vbe_b_t_map),
          fd_(fd) {}

    void operator()(const sycl::nd_item<1>& item) const;

 private:
    const at::PackedTensorAccessor32<int64_t, 1, RestrictPtrTraits>
        hash_size_cumsum_;
    const at::PackedTensorAccessor32<index_t, 1, RestrictPtrTraits> indices_;
    const at::PackedTensorAccessor32<index_t, 1, RestrictPtrTraits> offsets_;
    mutable at::PackedTensorAccessor32<info_acc_t, 1, RestrictPtrTraits> infos_;
    mutable at::PackedTensorAccessor32<index_t, 1, RestrictPtrTraits>
        linear_indices_;
    const int32_t info_B_num_bits_;
    const uint32_t info_B_mask_;
    const uint32_t max_T_;
    const uint32_t max_B_;
    const uint32_t* const __restrict__ vbe_b_t_map_;
    FixedDivisor fd_;
};

////////////////////////////////////////////////////////////////////////////////
// transpose_embedding_input - CSR to CSC Conversion
////////////////////////////////////////////////////////////////////////////////
//
// CUDA SOURCE MAPPING:
//   CUDA Function: transpose_embedding_input
//   CUDA File: transpose_embedding_input.cu
//   CUDA Path: FBGEMM/fbgemm_gpu/src/split_embeddings_utils/
//
// DESCRIPTION:
//   "Transpose" embedding inputs by sorting indices by their values.
//   Converts compressed sparse row (CSR) to compressed sparse column (CSC).

//
////////////////////////////////////////////////////////////////////////////////
std::tuple<Tensor /*linear_indices*/, Tensor /*linear_indices_sorted*/, \
           Tensor /*infos_sorted*/, Tensor /*sorted_linear_indices_run*/,  \
           Tensor /*sorted_linear_indices_run_lengths*/, \
           Tensor /*sorted_linear_indices_num_runs*/, \
           Tensor /*sorted_linear_indices_cumulative_run_lengths*/>
transpose_embedding_input(
    Tensor hash_size_cumsum, int64_t total_hash_size_bits, Tensor indices,
    Tensor offsets, bool nobag = false,
    const std::optional<Tensor>& vbe_b_t_map = std::optional<at::Tensor>(),
    const int64_t info_B_num_bits = 26, const int64_t info_B_mask = 0x2FFFFFF,
    const int64_t total_unique_indices = -1, const bool is_index_select = false,
    const std::optional<Tensor>& total_L_offsets = std::optional<at::Tensor>(),
    const int64_t fixed_L_per_warp = 0,
    const int64_t num_warps_per_feature = 0);

////////////////////////////////////////////////////////////////////////////////
// compute_grad_sum_unweighted_nobag - Gradient Accumulation Helper
////////////////////////////////////////////////////////////////////////////////
//
// CUDA SOURCE MAPPING:
//   CUDA Function: compute_grad_sum_unweighted_nobag
//   CUDA Generated File:
//   gen_embedding_backward_split_unweighted_nobag_device_kernel.cuh CUDA
//   Generated Path: fbgemm_gpu/_skbuild/linux-x86_64-3.10/cmake-build/ CUDA
//   Lines: 33-90 (complete implementation)
//
//   Template Source: embedding_backward_split_device_kernel_template.cuh
//   Template Path: FBGEMM/fbgemm_gpu/codegen/training/backward/
//
// DESCRIPTION:
//   Accumulates gradients from grad_output into local accumulator grad_sum.
//   Uses warp-level cooperation with sub-group shuffle operations for coalesced
//   access. Processes no-bag embeddings where each index maps to a single
//   output row.
//
//   This helper is called within backward kernels to compute the sum of
//   gradients for all occurrences of a specific embedding index.
//
////////////////////////////////////////////////////////////////////////////////
// Keep this template definition in the header so SYCL device compilation
// can instantiate it at kernel call sites across translation units.
template <typename grad_t, typename cache_t, int32_t kFixedMaxVecsPerThread, \
          int32_t kThreadGroupSize, int32_t VEC_WIDTH, bool kUseVecBlocking>
void compute_grad_sum_unweighted_nobag(
    const sycl::nd_item<2>& item, Vec4TAcc<cache_t>* grad_sum,
    Vec4TAcc<cache_t>* smem_grad_sum,
    const at::PackedTensorAccessor64<grad_t, 2, RestrictPtrTraits>& grad_output,
    const int32_t D, const int32_t T,
    const at::PackedTensorAccessor32<int64_t, 1, RestrictPtrTraits>&
        sorted_infos,
    const int32_t segment_start, const int32_t sl_start, const int32_t sl_end,
    const int32_t num_vecs) {
    const auto threadIdx_x = item.get_local_id(1);
    const auto sg = item.get_sub_group();

    // Copy value to vecs to make num_vecs known at compile time when
    // kUseVecBlocking == false
    const int32_t vecs = kUseVecBlocking ? num_vecs : kFixedMaxVecsPerThread;
    for (int32_t vec_start = 0; vec_start < vecs;
         vec_start += kFixedMaxVecsPerThread) {
// Reset grad_sum vectors
#pragma unroll kFixedMaxVecsPerThread
        for (int32_t vec = 0; vec < kFixedMaxVecsPerThread; vec++) {
            grad_sum[vec].acc.x() = 0;
            grad_sum[vec].acc.y() = 0;
            grad_sum[vec].acc.z() = 0;
            grad_sum[vec].acc.w() = 0;
        }

        for (int32_t sl = sl_start; sl < sl_end; sl += kThreadGroupSize) {
            auto sl_j = sl + threadIdx_x;  // if not nobag
            int64_t l_t =
                sl_j < sl_end ? sorted_infos[segment_start + sl_j] : 0;
            int32_t l = l_t / T;  // if not nobag
            for (int32_t j = 0; j < kThreadGroupSize && sl + j < sl_end; ++j) {
                int32_t l_j = sycl::select_from_group(sg, l, j);

#pragma unroll kFixedMaxVecsPerThread
                for (int32_t vec = 0;
                     vec < kFixedMaxVecsPerThread &&
                     (((vec + vec_start) * kThreadGroupSize + threadIdx_x) *
                      VEC_WIDTH) < D;
                     ++vec) {
                    const int32_t d =
                        (((vec + vec_start) * kThreadGroupSize + threadIdx_x) *
                         VEC_WIDTH);
                    Vec4TAcc<grad_t> grad_out_vec(
                        &grad_output[l_j][d]);  // if nobag
                    grad_sum[vec].add_(grad_out_vec);
                }
            }
        }

        if (smem_grad_sum) {
// Store grad_sum in smem_grad_sum
#pragma unroll kFixedMaxVecsPerThread
            for (int32_t vec = 0;
                 (vec < kFixedMaxVecsPerThread) &&
                 ((vec + vec_start) * kThreadGroupSize + threadIdx_x) *
                         VEC_WIDTH <
                     D;
                 ++vec) {
                const int32_t d_vec =
                    ((vec + vec_start) * kThreadGroupSize + threadIdx_x);
                smem_grad_sum[d_vec] = grad_sum[vec];
            }
        }
    }
}
}  // namespace fbgemm_xpu
