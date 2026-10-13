/*
 * Copyright (c) Meta Platforms, Inc. and affiliates. All rights reserved.
 * Copyright (c) 2026 Intel Corporation. All Rights Reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */

////////////////////////////////////////////////////////////////////////////////
// SYCL PORT MAPPING TO FBGEMM CUDA SOURCE - SHARED JAGGED TENSOR INFRASTRUCTURE
////////////////////////////////////////////////////////////////////////////////
//
// Shared SYCL infrastructure used by every jagged tensor operator in this
// directory. SYCL counterpart of FBGEMM's jagged_tensor_ops/common.cuh and
// jagged_tensor_ops/common.h.
//
// ORIGINAL CUDA SOURCE:
//   File: fbgemm_gpu/src/jagged_tensor_ops/common.cuh
//   File: fbgemm_gpu/src/jagged_tensor_ops/common.h
//
// STRUCT MAPPING:
//   StackArray<T> (SYCL)
//     → StackArray<T> (CUDA)
//   HalfVec8 / HalfVec4 / HalfVec2 (SYCL)
//     → VecType128 / VecType64 / VecType32 (CUDA)
//
// DEVICE HELPER MAPPING:
//   walk_down_tensor_storage_tree_ (SYCL)
//     → walk_down_tensor_storage_tree_ (CUDA)
//   apply_vec / apply_scalar (SYCL)
//     → f128 / f64 / f32 / fh (CUDA)
//
// KERNEL MAPPING:
//   JaggedDenseElementwiseDenseOutputKernel (SYCL)
//     → jagged_dense_elementwise_dense_output_kernel_ (CUDA)
//
//   JaggedDenseDenseElementwiseJaggedOutputKernel (SYCL)
//     → jagged_dense_dense_elementwise_jagged_output_kernel_ (CUDA)
//
//   JaggedDenseDenseElementwiseJaggedOutputOptSearchKernel (SYCL)
//     → jagged_dense_dense_elementwise_jagged_output_opt_search_kernel_ (CUDA)
//
//   JaggedDenseDenseElementwiseJaggedOutputOptGatherKernel (SYCL)
//     → jagged_dense_dense_elementwise_jagged_output_opt_gather_kernel_ (CUDA)
//
// HOST FUNCTION MAPPING:
//   check_shape_and_partition_ (SYCL)
//     → check_shape_and_partition_ (CUDA)
//   jagged_dense_elementwise_dense_output_ (SYCL)
//     → jagged_dense_elementwise_dense_output_ (CUDA)
//   jagged_dense_elementwise_jagged_output_ (SYCL)
//     → jagged_dense_elementwise_jagged_output_ (CUDA)
//   jagged_dense_elementwise_jagged_output_opt_ (SYCL)
//     → jagged_dense_elementwise_jagged_output_opt_ (CUDA)
//   jagged_dense_dense_elementwise_jagged_output_matches_opt (SYCL)
//     → jagged_dense_dense_elementwise_jagged_output_matches_opt (CUDA)
//
////////////////////////////////////////////////////////////////////////////////

#pragma once

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <limits>
#include <vector>

#include <sycl/sycl.hpp>

#include <c10/xpu/XPUStream.h>

#include <ATen/ATen.h>
#include <ATen/Dispatch.h>
#include <ATen/native/StridedRandomAccessor.h>
#include <torch/library.h>

#include "fbgemm_utils/dispatch_macros.h"
#include "fbgemm_utils/utils.h"
#include "fbgemm_utils/tensor_utils.h"

using at::native::RestrictPtrTraits;

namespace fbgemm_xpu {

// ============================================================================
// Constants and small helpers
// ============================================================================

// Maximum number of jagged dimensions carried through a StackArray.
// CUDA source: kStackArrayMaxDims in common.cuh.
constexpr size_t kStackArrayMaxDims = 5;

// CUDA caps grid.y at 65535 for the _opt_ gather kernel; mirrored here.
constexpr int64_t kJaggedMaxGroups = 65535;

// CUDA SOURCE MAPPING:
//   CUDA Struct: StackArray<T>
//   CUDA File: fbgemm_gpu/src/jagged_tensor_ops/common.cuh

template <typename T>
struct StackArray {
    T vals[kStackArrayMaxDims];
    size_t ndim;
};

// ============================================================================
// Element-wise operation functors
// ============================================================================
//
// CUDA SOURCE MAPPING:
//   CUDA passes `[] __device__ (scalar_t x, scalar_t y) -> scalar_t` lambdas as
//   template parameter F, and wraps each in a 3-argument `ff` at the
//   jagged-output launch sites. SYCL kernels must be functors, so each CUDA
//   lambda becomes a named struct here providing both arities directly.
//
// Arithmetic is performed in float and rounded once on store - deviation (1).
//
// f = x  (jagged_to_padded_dense_forward)
struct JaggedOpCopyX {
    template <typename T>
    inline T operator()(T x, T /*y_0*/) const {
        return x;
    }
    template <typename T>
    inline T operator()(T x, T /*y_0*/, T /*y_1*/) const {
        return x;
    }
};

// f = y  (jagged_to_padded_dense_backward, dense_to_jagged_forward)
struct JaggedOpCopyY {
    template <typename T>
    inline T operator()(T /*x*/, T y_0) const {
        return y_0;
    }
    template <typename T>
    inline T operator()(T /*x*/, T y_0, T /*y_1*/) const {
        return y_0;
    }
};

// f = x + y  (jagged_dense_elementwise_add_jagged_output)
struct JaggedOpAdd {
    template <typename T>
    inline T operator()(T x, T y_0) const {
        return static_cast<T>(static_cast<float>(x) + static_cast<float>(y_0));
    }
    template <typename T>
    inline T operator()(T x, T y_0, T /*y_1*/) const {
        return static_cast<T>(static_cast<float>(x) + static_cast<float>(y_0));
    }
};

// f = x * y  (jagged_dense_elementwise_mul_forward/backward)
struct JaggedOpMul {
    template <typename T>
    inline T operator()(T x, T y_0) const {
        return static_cast<T>(static_cast<float>(x) * static_cast<float>(y_0));
    }
    template <typename T>
    inline T operator()(T x, T y_0, T /*y_1*/) const {
        return static_cast<T>(static_cast<float>(x) * static_cast<float>(y_0));
    }
};

// ============================================================================
// walk_down_tensor_storage_tree_ - jagged storage tree traversal
// ============================================================================
//
// CUDA SOURCE MAPPING:
//   CUDA Function: walk_down_tensor_storage_tree_
//   CUDA File: fbgemm_gpu/src/jagged_tensor_ops/common.cuh
//

template <int NUM_JAGGED_DIM, typename PosT, typename index_t>
inline bool walk_down_tensor_storage_tree_(
    PosT& offset,
    const PosT flattened_jagged_idx,
    const StackArray<int64_t>& jagged_dims,
    const StackArray<const index_t*>& x_offsets) {
    // Compute coordinates.
    PosT jagged_coords[NUM_JAGGED_DIM];
    PosT j_temp = flattened_jagged_idx;
#pragma unroll
    for (int d = NUM_JAGGED_DIM - 1; d >= 0; --d) {
        const PosT jagged_size = jagged_dims.vals[d];
        jagged_coords[d] = j_temp % jagged_size;
        j_temp /= jagged_size;
    }

    // Walk down the tree.
    bool is_zero = false;
#pragma unroll
    for (int d = 0; d < NUM_JAGGED_DIM; ++d) {
        const PosT begin = x_offsets.vals[d][offset];
        const PosT end = x_offsets.vals[d][offset + 1];
        if (jagged_coords[d] >= end - begin) {
            is_zero = true;
            break;
        }
        offset = begin + jagged_coords[d];
    }
    return is_zero;
}

////////////////////////////////////////////////////////////////////////////////
// JaggedDenseElementwiseDenseOutputKernel - Device Kernel
////////////////////////////////////////////////////////////////////////////////
//
// CUDA SOURCE MAPPING:
//   CUDA Kernel: jagged_dense_elementwise_dense_output_kernel_
//   CUDA File: fbgemm_gpu/src/jagged_tensor_ops/common.cuh
//

template <
    int NUM_JAGGED_DIM,
    typename IdxT,
    typename index_t,
    typename scalar_t,
    typename F>
class JaggedDenseElementwiseDenseOutputKernel {
public:
    JaggedDenseElementwiseDenseOutputKernel(
        at::GenericPackedTensorAccessor<scalar_t, 2, RestrictPtrTraits, IdxT>
            x_values,
        StackArray<const index_t*> x_offsets,
        at::GenericPackedTensorAccessor<scalar_t, 3, RestrictPtrTraits, IdxT> y,
        at::GenericPackedTensorAccessor<scalar_t, 3, RestrictPtrTraits, IdxT>
            output,
        StackArray<int64_t> jagged_dims,
        F f,
        scalar_t padding_value)
        : x_values_(x_values),
          x_offsets_(x_offsets),
          y_(y),
          output_(output),
          jagged_dims_(jagged_dims),
          f_(f),
          padding_value_(padding_value) {}

    void operator()(const sycl::nd_item<2>& item) const {
        const IdxT outer_dense_size = y_.size(0);
        const IdxT jagged_folded_size = y_.size(1);
        const IdxT inner_dense_size = y_.size(2);

        // CUDA: blockIdx.x * blockDim.y + threadIdx.y
        const IdxT outer_begin = static_cast<IdxT>(
            item.get_group(0) * item.get_local_range(0) + item.get_local_id(0));
        // CUDA: gridDim.x * blockDim.y
        const IdxT outer_stride = static_cast<IdxT>(
            item.get_group_range(0) * item.get_local_range(0));
        const IdxT total_outer = outer_dense_size * jagged_folded_size;

        // CUDA: threadIdx.x / blockDim.x
        const IdxT inner_begin = static_cast<IdxT>(item.get_local_id(1));
        const IdxT inner_stride = static_cast<IdxT>(item.get_local_range(1));

        // The inner loops below step over element pairs. Testing
        // `iidx * 2 + 1 < inner_dense_size` evaluates that product for the
        // first iidx that fails the test, which passes INT32_MAX once
        // inner_dense_size approaches the int32 limit. Comparing against the
        // pair count is equivalent for both parities - for inner_dense_size
        // 2h and 2h + 1 alike the original test holds exactly while iidx < h -
        // and keeps every term in range.
        const IdxT inner_pairs = inner_dense_size / 2;
        const bool has_odd_tail = (inner_dense_size & 1) != 0;

        for (IdxT outer = outer_begin; outer < total_outer;) {
            const IdxT oidx = outer / jagged_folded_size;
            const IdxT jidx = outer % jagged_folded_size;

            IdxT offset = oidx;
            const bool is_zero = walk_down_tensor_storage_tree_<NUM_JAGGED_DIM>(
                offset, jidx, jagged_dims_, x_offsets_);

            if (is_zero) {
                IdxT iidx;
                for (iidx = inner_begin; iidx < inner_pairs;
                     iidx += inner_stride) {
                    output_[oidx][jidx][2 * iidx] =
                        f_(padding_value_, y_[oidx][jidx][2 * iidx]);
                    output_[oidx][jidx][2 * iidx + 1] =
                        f_(padding_value_, y_[oidx][jidx][2 * iidx + 1]);
                }
                if (has_odd_tail && iidx == inner_pairs) {
                    output_[oidx][jidx][2 * iidx] =
                        f_(padding_value_, y_[oidx][jidx][2 * iidx]);
                }
            } else {
                IdxT iidx;
                for (iidx = inner_begin; iidx < inner_pairs;
                     iidx += inner_stride) {
                    output_[oidx][jidx][2 * iidx] = f_(
                        x_values_[offset][2 * iidx], y_[oidx][jidx][2 * iidx]);
                    output_[oidx][jidx][2 * iidx + 1] = f_(
                        x_values_[offset][2 * iidx + 1],
                        y_[oidx][jidx][2 * iidx + 1]);
                }
                if (has_odd_tail && iidx == inner_pairs) {
                    output_[oidx][jidx][2 * iidx] = f_(
                        x_values_[offset][2 * iidx], y_[oidx][jidx][2 * iidx]);
                }
            }

            // `total_outer - outer` is in [1, total_outer] so it cannot
            // overflow, unlike `outer + outer_stride`, which passes INT32_MAX
            // on the last active work-item when total_outer approaches the
            // int32 limit: the numel < INT32_MAX gate bounds the indices this
            // kernel forms, not the one-past-the-end value of this counter.
            if (total_outer - outer <= outer_stride) {
                break;
            }
            outer += outer_stride;
        }
    }

private:
    at::GenericPackedTensorAccessor<scalar_t, 2, RestrictPtrTraits, IdxT>
        x_values_;
    StackArray<const index_t*> x_offsets_;
    at::GenericPackedTensorAccessor<scalar_t, 3, RestrictPtrTraits, IdxT> y_;
    // Written by the kernel, so it must stay assignable inside operator() const.
    mutable at::GenericPackedTensorAccessor<scalar_t, 3, RestrictPtrTraits, IdxT>
        output_;
    StackArray<int64_t> jagged_dims_;
    F f_;
    scalar_t padding_value_;
};

////////////////////////////////////////////////////////////////////////////////
// JaggedDenseDenseElementwiseJaggedOutputKernel - Device Kernel
////////////////////////////////////////////////////////////////////////////////
//
// CUDA SOURCE MAPPING:
//   CUDA Kernel: jagged_dense_dense_elementwise_jagged_output_kernel_
//   CUDA File: fbgemm_gpu/src/jagged_tensor_ops/common.cuh
//

template <
    int NUM_JAGGED_DIM,
    typename index_t,
    typename scalar_t,
    typename F>
class JaggedDenseDenseElementwiseJaggedOutputKernel {
public:
    JaggedDenseDenseElementwiseJaggedOutputKernel(
        at::PackedTensorAccessor32<scalar_t, 2, RestrictPtrTraits> x_values,
        StackArray<const index_t*> x_offsets,
        StackArray<int64_t> x_offsets_sizes,
        at::PackedTensorAccessor32<scalar_t, 3, RestrictPtrTraits> y_0,
        at::PackedTensorAccessor32<scalar_t, 3, RestrictPtrTraits> y_1,
        at::PackedTensorAccessor32<scalar_t, 2, RestrictPtrTraits> output_values,
        StackArray<int64_t> jagged_dims,
        F f)
        : x_values_(x_values),
          x_offsets_(x_offsets),
          x_offsets_sizes_(x_offsets_sizes),
          y_0_(y_0),
          y_1_(y_1),
          output_values_(output_values),
          jagged_dims_(jagged_dims),
          f_(f) {}

    void operator()(const sycl::nd_item<2>& item) const {
        const int outer_dense_size = y_0_.size(0);
        const int inner_dense_size = y_0_.size(2);
        const int nnz = x_values_.size(0);

        // CUDA: blockIdx.x * blockDim.y + threadIdx.y
        const int offset_begin = static_cast<int>(
            item.get_group(0) * item.get_local_range(0) + item.get_local_id(0));
        // CUDA: gridDim.x * blockDim.y
        const int offset_stride = static_cast<int>(
            item.get_group_range(0) * item.get_local_range(0));

        // CUDA: threadIdx.x / blockDim.x
        const int inner_begin = static_cast<int>(item.get_local_id(1));
        const int inner_stride = static_cast<int>(item.get_local_range(1));

        // Pair count instead of `iidx * 2 + 1 < inner_dense_size`; see the
        // matching comment in JaggedDenseElementwiseDenseOutputKernel.
        const int inner_pairs = inner_dense_size / 2;
        const bool has_odd_tail = (inner_dense_size & 1) != 0;

        for (int offset = offset_begin; offset < nnz;) {
            int offset_temp = offset;
            int jidx = 0;
            bool truncated = false;
            int dim_prod = 1;
#pragma unroll
            for (int d = NUM_JAGGED_DIM - 1; d >= 0; --d) {
                // Binary search the first that is bigger than offset.
                int count = x_offsets_sizes_.vals[d] - 1;
                int first = 1;
                while (count > 0) {
                    int idx = first;
                    int step = count / 2;
                    idx += step;
                    if (x_offsets_.vals[d][idx] <= offset_temp) {
                        first = ++idx;
                        count -= step + 1;
                    } else {
                        count = step;
                    }
                }

                --first;
                const int coord = offset_temp - x_offsets_.vals[d][first];
                if (coord >= jagged_dims_.vals[d]) {
                    truncated = true;
                    break;
                }
                jidx += coord * dim_prod;
                dim_prod *= jagged_dims_.vals[d];
                offset_temp = first;
            }

            if (offset_temp >= outer_dense_size) {
                // Can happen when values has more elements than the last
                // element of offsets.
                truncated = true;
            }

            if (!truncated) {
                const int oidx = offset_temp;
                int iidx;
                for (iidx = inner_begin; iidx < inner_pairs;
                     iidx += inner_stride) {
                    output_values_[offset][2 * iidx] = f_(
                        x_values_[offset][2 * iidx],
                        y_0_[oidx][jidx][2 * iidx],
                        y_1_[oidx][jidx][2 * iidx]);
                    output_values_[offset][2 * iidx + 1] = f_(
                        x_values_[offset][2 * iidx + 1],
                        y_0_[oidx][jidx][2 * iidx + 1],
                        y_1_[oidx][jidx][2 * iidx + 1]);
                }
                if (has_odd_tail && iidx == inner_pairs) {
                    output_values_[offset][2 * iidx] = f_(
                        x_values_[offset][2 * iidx],
                        y_0_[oidx][jidx][2 * iidx],
                        y_1_[oidx][jidx][2 * iidx]);
                }
            } else {
                int iidx;
                for (iidx = inner_begin; iidx < inner_pairs;
                     iidx += inner_stride) {
                    output_values_[offset][2 * iidx] = f_(
                        x_values_[offset][2 * iidx], scalar_t(0), scalar_t(0));
                    output_values_[offset][2 * iidx + 1] = f_(
                        x_values_[offset][2 * iidx + 1],
                        scalar_t(0),
                        scalar_t(0));
                }
                if (has_odd_tail && iidx == inner_pairs) {
                    output_values_[offset][2 * iidx] = f_(
                        x_values_[offset][2 * iidx], scalar_t(0), scalar_t(0));
                }
            }

            // Guarded increment; see the matching comment in
            // JaggedDenseElementwiseDenseOutputKernel. `nnz` is bounded by
            // x_values.numel(), but `offset + offset_stride` is not.
            if (nnz - offset <= offset_stride) {
                break;
            }
            offset += offset_stride;
        }
    }

private:
    at::PackedTensorAccessor32<scalar_t, 2, RestrictPtrTraits> x_values_;
    StackArray<const index_t*> x_offsets_;
    StackArray<int64_t> x_offsets_sizes_;
    at::PackedTensorAccessor32<scalar_t, 3, RestrictPtrTraits> y_0_;
    at::PackedTensorAccessor32<scalar_t, 3, RestrictPtrTraits> y_1_;
    // Written by the kernel, so it must stay assignable inside operator() const.
    mutable at::PackedTensorAccessor32<scalar_t, 2, RestrictPtrTraits>
        output_values_;
    StackArray<int64_t> jagged_dims_;
    F f_;
};

inline void check_tensors_on_same_xpu_(
    const at::Tensor& x_values,
    const std::vector<at::Tensor>& x_offsets,
    const at::Tensor& y,
    const at::Tensor& output) {
    TENSOR_ON_SYCL_XPU(x_values);
    TENSORS_ON_SAME_SYCL_XPU_IF_NOT_OPTIONAL(x_values, y, output);
    for (const auto& x_offset : x_offsets) {
        TENSORS_ON_SAME_SYCL_XPU_IF_NOT_OPTIONAL(x_values, x_offset);
    }
}

// ============================================================================
// check_shape_and_partition_ - launch geometry + jagged dim extraction
// ============================================================================
//
// CUDA SOURCE MAPPING:
//   CUDA Function: check_shape_and_partition_
//   CUDA File: fbgemm_gpu/src/jagged_tensor_ops/common.cuh
//

struct JaggedLaunchConfig {
    int64_t threads_x;  // CUDA threads.x - folded inner dense dim
    int64_t threads_y;  // CUDA threads.y - rows per work-group
    int64_t blocks;     // CUDA blocks.x
    StackArray<int64_t> jagged_dims;

    inline sycl::nd_range<2> nd_range() const {
        return sycl::nd_range<2>(
            sycl::range<2>(blocks * threads_y, threads_x),
            sycl::range<2>(threads_y, threads_x));
    }
};

inline void check_jagged_dense_shape_(
    const at::Tensor& values,
    const std::vector<at::Tensor>& offsets,
    const at::Tensor& dense_tensor) {
    const int64_t outer_dense_size = dense_tensor.size(0);
    TORCH_CHECK(
        outer_dense_size == offsets[0].numel() - 1,
        "outer_dense_size, ",
        outer_dense_size,
        " != offsets[0].numel() - 1, ",
        offsets[0].numel() - 1);
    const int64_t inner_dense_size = dense_tensor.size(-1);
    TORCH_CHECK(
        inner_dense_size == values.size(-1),
        "inner_dense_size, ",
        inner_dense_size,
        " != values.size(-1), ",
        values.size(-1));
}

inline JaggedLaunchConfig check_shape_and_partition_(
    const at::Tensor& values,
    const std::vector<at::Tensor>& offsets,
    const at::Tensor& dense_tensor) {
    check_jagged_dense_shape_(values, offsets, dense_tensor);

    const int64_t outer_dense_size = dense_tensor.size(0);
    const int64_t inner_dense_size = dense_tensor.size(-1);
    const int64_t jagged_folded_size =
        dense_tensor.numel() / (outer_dense_size * inner_dense_size);

    constexpr int64_t kWarpSize = static_cast<int64_t>(kThreadGroupSize);

    // DIVERGENCE FROM CUDA, which uses
    //   inner_dense_size >= kWarpSize / 2 ? kWarpSize : inner_dense_size.
    // That threshold accounts for the two-elements-per-work-item inner loops in
    // all three consuming kernels, but the value does not: they stride over
    // inner_dense_size / 2 pairs, so the inner axis needs only
    // ceil(inner_dense_size / 2) lanes. CUDA's value over-provisions for every
    // inner_dense_size < 2 * kWarpSize, worst at exactly kWarpSize / 2, where the
    // ternary jumps from 15 lanes to 32 while the pair count only goes 7 -> 8:
    // 24 of 32 inner lanes fail their first loop test. Those idle work-items hold
    // Xe-core thread slots for the whole work-group's lifetime, where CUDA retires
    // empty warps at issue - on B60, BF16 inner_dense_size 16 measured 3.6-3.8x
    // slower than the pair-derived extent below.
    //
    // Only the clamp to 1 is load-bearing for correctness: a zero-width inner dim
    // would otherwise produce an empty nd_range (CUDA tolerates a zero-sized
    // block dim by never launching). Every other extent is just a loop stride -
    // the lane at inner_pairs % threads_x always exits exactly on the pair count,
    // so the odd tail element at 2 * (inner_dense_size / 2) is written for any
    // extent >= 1. Rounding the pair count up rather than down is therefore a
    // throughput choice: it gives odd widths one more lane and saves that lane a
    // second trip round the loop.
    const int64_t threads_x = std::max<int64_t>(
        1, std::min<int64_t>(kWarpSize, div_round_up(inner_dense_size, 2)));
    const int64_t threads_y = static_cast<int64_t>(kMaxThreads) / kWarpSize;

    // The work-group count is capped so the flattened work-item
    // count of the launch fits in an int32_t. CUDA leaves blocks.x uncapped
    // because its only limit is gridDim.x <= 2^31-1, which the same expression
    // cannot exceed in practice. DPC++ is stricter: it assumes by default
    // (-fsycl-id-queries-fit-in-int) that the TOTAL work-item count fits in an
    // int, and queue::submit throws once blocks * threads_y * threads_x passes
    // INT32_MAX. Because threads_x <= inner_dense_size above, that product is at
    // most the dense operand's numel plus the final work-group round-up. The
    // generic dense-output consumer accepts an over-limit numel through its
    // packed_accessor64 fallback. The jagged-jagged dense-output consumer uses
    // packed_accessor32, but at D = 1 the round-up can still push a legal numel
    // over the launch limit.
    //
    // Capping is safe because all three consumers of this config iterate their
    // outer index with a group-stride loop, so a smaller launch still covers
    // every row. The jagged-output consumer re-derives blocks from nnz rather
    // than from outer * folded, so it must re-apply the same cap itself - see
    // FBGEMM_XPU_JAGGED_OUTPUT_INVOKE_BODY. It builds packed_accessor32
    // unconditionally, but the D = 1 work-group round-up can likewise reach the
    // cap while numel remains legal.
    const int64_t blocks = std::max<int64_t>(
        1,
        static_cast<int64_t>(xpu_cap_grid_dim_x(
            div_round_up(outer_dense_size * jagged_folded_size, threads_y),
            threads_x * threads_y)));

    StackArray<int64_t> jagged_dims_tensor;
    const int num_jagged_dim = dense_tensor.dim() - 2;
    TORCH_CHECK(num_jagged_dim <= static_cast<int>(kStackArrayMaxDims));
    jagged_dims_tensor.ndim = num_jagged_dim;
    std::memcpy(
        &(jagged_dims_tensor.vals[0]),
        dense_tensor.sizes().data() + 1,
        num_jagged_dim * sizeof(int64_t));

    return JaggedLaunchConfig{
        threads_x, threads_y, blocks, jagged_dims_tensor};
}

// Collect contiguous copies of the offsets tensors plus their device pointers.
// The returned vector owns the copies and must outlive the kernel launch.
template <typename index_t>
inline std::vector<at::Tensor> collect_offsets_(
    const std::vector<at::Tensor>& x_offsets,
    int num_jagged_dim,
    StackArray<const index_t*>& x_offset_ptrs,
    StackArray<int64_t>& x_offset_sizes) {
    std::vector<at::Tensor> x_offsets_contig(num_jagged_dim);
    x_offset_ptrs.ndim = num_jagged_dim;
    x_offset_sizes.ndim = num_jagged_dim;
    for (int d = 0; d < num_jagged_dim; ++d) {
        x_offsets_contig[d] = x_offsets[d].contiguous();
        x_offset_ptrs.vals[d] = x_offsets_contig[d].data_ptr<index_t>();
        x_offset_sizes.vals[d] = x_offsets[d].numel();
    }
    return x_offsets_contig;
}

// ============================================================================
// FBGEMM_XPU_JAGGED_DISPATCH_DIMS - dispatch on the number of jagged dims
// ============================================================================
//
// CUDA SOURCE MAPPING:
//   CUDA Macro: JAGGED_TENSOR_DISPATCH_DIMS
//   CUDA File: fbgemm_gpu/src/jagged_tensor_ops/common.h
//

#define FBGEMM_XPU_JAGGED_DISPATCH_DIMS()                                     \
    AT_DISPATCH_INDEX_TYPES(                                                  \
        x_offsets[0].scalar_type(), "jagged_indices", [&] {                    \
            switch (num_jagged_dim) {                                         \
                case 1:                                                       \
                    INVOKE_KERNEL_WITH_DIM(1);                                \
                    break;                                                    \
                case 2:                                                       \
                    INVOKE_KERNEL_WITH_DIM(2);                                \
                    break;                                                    \
                case 3:                                                       \
                    INVOKE_KERNEL_WITH_DIM(3);                                \
                    break;                                                    \
                case 4:                                                       \
                    INVOKE_KERNEL_WITH_DIM(4);                                \
                    break;                                                    \
                case 5:                                                       \
                    INVOKE_KERNEL_WITH_DIM(5);                                \
                    break;                                                    \
                default:                                                      \
                    TORCH_CHECK(                                              \
                        false,                                                \
                        "unsupported number of jagged dim ",                  \
                        num_jagged_dim);                                      \
            }                                                                 \
        });

// ============================================================================
// jagged_dense_elementwise_dense_output_ - Host Function (jagged -> dense)
// ============================================================================
//
// CUDA SOURCE MAPPING:
//   CUDA Function: jagged_dense_elementwise_dense_output_
//   CUDA File: fbgemm_gpu/src/jagged_tensor_ops/common.cuh
//

template <typename scalar_t, typename F>
void jagged_dense_elementwise_dense_output_(
    const at::Tensor& x_values,
    const std::vector<at::Tensor>& x_offsets,
    const at::Tensor& y,
    const at::Tensor& output,
    F f,
    const scalar_t padding_value = static_cast<scalar_t>(0)) {
    check_tensors_on_same_xpu_(x_values, x_offsets, y, output);

    const int num_jagged_dim = y.dim() - 2;
    TORCH_CHECK(
        x_offsets.size() == static_cast<size_t>(num_jagged_dim),
        "x_offsets.size(), ",
        x_offsets.size(),
        " != num_jagged_dim ",
        num_jagged_dim);

    if (y.numel() == 0) {
        return;
    }

    const JaggedLaunchConfig cfg =
        check_shape_and_partition_(x_values, x_offsets, y);

    // Canonicalize y and output to 3D, collapsing jagged dimensions.
    const at::Tensor y_reshaped = y.view({y.size(0), -1, y.size(-1)});
    const at::Tensor output_reshaped = output.view(y_reshaped.sizes());

    sycl::queue& queue = c10::xpu::getCurrentXPUStream().queue();

#define INVOKE_KERNEL_WITH_DIM(NUM_JAGGED_DIM)                                 \
    {                                                                          \
        StackArray<const index_t*> x_offset_ptrs;                              \
        StackArray<int64_t> x_offset_sizes;                                    \
        const auto x_offsets_contig = collect_offsets_<index_t>(               \
            x_offsets, num_jagged_dim, x_offset_ptrs, x_offset_sizes);         \
                                                                               \
        /* Pick int32 indexing when every touched tensor's numel fits in        \
         * int32; otherwise dispatch the int64 specialization. The int64 path   \
         * defends against `oidx * strides_[0]` overflowing int32 inside a      \
         * 32-bit accessor, which scribbles outside the buffer for high oidx    \
         * and surfaces as silent NaN downstream (T264042859). */               \
        constexpr int64_t kInt32Limit =                                        \
            static_cast<int64_t>(std::numeric_limits<int32_t>::max());         \
        const bool use_int32_indexing = x_values.numel() < kInt32Limit &&      \
            y_reshaped.numel() < kInt32Limit &&                                \
            output_reshaped.numel() < kInt32Limit;                             \
                                                                               \
        if (use_int32_indexing) {                                              \
            using KernelT = JaggedDenseElementwiseDenseOutputKernel<           \
                NUM_JAGGED_DIM,                                                \
                int32_t,                                                       \
                index_t,                                                       \
                scalar_t,                                                      \
                F>;                                                            \
            queue.submit([&](sycl::handler& cgh) {                             \
                cgh.parallel_for<KernelT>(                                     \
                    cfg.nd_range(),                                            \
                    KernelT(                                                   \
                        x_values.packed_accessor32<                            \
                            scalar_t,                                          \
                            2,                                                 \
                            RestrictPtrTraits>(),                              \
                        x_offset_ptrs,                                         \
                        y_reshaped.packed_accessor32<                          \
                            scalar_t,                                          \
                            3,                                                 \
                            RestrictPtrTraits>(),                              \
                        output_reshaped.packed_accessor32<                     \
                            scalar_t,                                          \
                            3,                                                 \
                            RestrictPtrTraits>(),                              \
                        cfg.jagged_dims,                                       \
                        f,                                                     \
                        padding_value));                                       \
            });                                                                \
        } else {                                                               \
            using KernelT = JaggedDenseElementwiseDenseOutputKernel<           \
                NUM_JAGGED_DIM,                                                \
                int64_t,                                                       \
                index_t,                                                       \
                scalar_t,                                                      \
                F>;                                                            \
            queue.submit([&](sycl::handler& cgh) {                             \
                cgh.parallel_for<KernelT>(                                     \
                    cfg.nd_range(),                                            \
                    KernelT(                                                   \
                        x_values.packed_accessor64<                            \
                            scalar_t,                                          \
                            2,                                                 \
                            RestrictPtrTraits>(),                              \
                        x_offset_ptrs,                                         \
                        y_reshaped.packed_accessor64<                          \
                            scalar_t,                                          \
                            3,                                                 \
                            RestrictPtrTraits>(),                              \
                        output_reshaped.packed_accessor64<                     \
                            scalar_t,                                          \
                            3,                                                 \
                            RestrictPtrTraits>(),                              \
                        cfg.jagged_dims,                                       \
                        f,                                                     \
                        padding_value));                                       \
            });                                                                \
        }                                                                      \
    }

    FBGEMM_XPU_JAGGED_DISPATCH_DIMS();

#undef INVOKE_KERNEL_WITH_DIM
}

// Overload allocating the output, mirroring the CUDA convenience overload.
template <typename scalar_t, typename F>
at::Tensor jagged_dense_elementwise_dense_output_(
    const at::Tensor& x_values,
    const std::vector<at::Tensor>& x_offsets,
    const at::Tensor& y,
    F f,
    const scalar_t padding_value = static_cast<scalar_t>(0)) {
    at::Tensor output = at::empty_like(y);
    jagged_dense_elementwise_dense_output_(
        x_values, x_offsets, y, output, f, padding_value);
    return output;
}

// ============================================================================
// FBGEMM_XPU_JAGGED_OUTPUT_INVOKE_BODY - generic jagged-output launch
// ============================================================================
//
// CUDA SOURCE MAPPING:
//   CUDA Macro: INVOKE_KERNEL_WITH_DIM (jagged-output variant)
//   CUDA File: fbgemm_gpu/src/jagged_tensor_ops/common.cuh
//

#define FBGEMM_XPU_JAGGED_OUTPUT_INVOKE_BODY(NUM_JAGGED_DIM)                   \
    {                                                                          \
        JaggedLaunchConfig cfg =                                               \
            check_shape_and_partition_(x_values, x_offsets, y);                \
        /* This kernel strides over nnz, not outer * folded, so the grid is     \
         * re-derived here. Re-apply the int32 launch-size cap that             \
         * check_shape_and_partition_ applied to the value being replaced;      \
         * the group-stride loop over `offset` still covers every row. */       \
        cfg.blocks = std::max<int64_t>(                                        \
            1,                                                                 \
            static_cast<int64_t>(xpu_cap_grid_dim_x(                           \
                div_round_up(x_values.size(0), cfg.threads_y),                 \
                cfg.threads_x * cfg.threads_y)));                              \
                                                                               \
        StackArray<const index_t*> x_offset_ptrs;                             \
        StackArray<int64_t> x_offset_sizes;                                    \
        const auto x_offsets_contig = collect_offsets_<index_t>(               \
            x_offsets, num_jagged_dim, x_offset_ptrs, x_offset_sizes);         \
                                                                               \
        using KernelT = JaggedDenseDenseElementwiseJaggedOutputKernel<         \
            NUM_JAGGED_DIM,                                                    \
            index_t,                                                           \
            scalar_t,                                                          \
            F>;                                                                \
        queue.submit([&](sycl::handler& cgh) {                                 \
            cgh.parallel_for<KernelT>(                                         \
                cfg.nd_range(),                                                \
                KernelT(                                                       \
                    x_values                                                   \
                        .packed_accessor32<scalar_t, 2, RestrictPtrTraits>(),   \
                    x_offset_ptrs,                                             \
                    x_offset_sizes,                                            \
                    y_reshaped                                                 \
                        .packed_accessor32<scalar_t, 3, RestrictPtrTraits>(),   \
                    y_reshaped                                                 \
                        .packed_accessor32<scalar_t, 3, RestrictPtrTraits>(),   \
                    output_values                                              \
                        .packed_accessor32<scalar_t, 2, RestrictPtrTraits>(),   \
                    cfg.jagged_dims,                                           \
                    f));                                                       \
        });                                                                    \
    }

// ============================================================================
// jagged_dense_elementwise_jagged_output_ - Host Function (dense -> jagged)
// ============================================================================
//
// CUDA SOURCE MAPPING:
//   CUDA Function: jagged_dense_elementwise_jagged_output_
//   CUDA File: fbgemm_gpu/src/jagged_tensor_ops/common.cuh
//

template <typename scalar_t, typename F>
void jagged_dense_elementwise_jagged_output_(
    const at::Tensor& x_values,
    const std::vector<at::Tensor>& x_offsets,
    const at::Tensor& y,
    const at::Tensor& output_values,
    F f) {
    check_tensors_on_same_xpu_(x_values, x_offsets, y, output_values);

    const int num_jagged_dim = y.dim() - 2;
    TORCH_CHECK(
        x_offsets.size() == static_cast<size_t>(num_jagged_dim),
        "x_offsets.size(), ",
        x_offsets.size(),
        " != num_jagged_dim, ",
        num_jagged_dim);

    if (y.numel() == 0 || x_values.numel() == 0) {
        return;
    }

    // Canonicalize y to 3D, collapsing jagged dimensions.
    const at::Tensor y_reshaped = y.view({y.size(0), -1, y.size(-1)});

    sycl::queue& queue = c10::xpu::getCurrentXPUStream().queue();

#define INVOKE_KERNEL_WITH_DIM(NUM_JAGGED_DIM) \
    FBGEMM_XPU_JAGGED_OUTPUT_INVOKE_BODY(NUM_JAGGED_DIM)

    FBGEMM_XPU_JAGGED_DISPATCH_DIMS();

#undef INVOKE_KERNEL_WITH_DIM
}

////////////////////////////////////////////////////////////////////////////////
// HALF-PRECISION VECTORIZED FAST PATH (_opt_)
////////////////////////////////////////////////////////////////////////////////

////////////////////////////////////////////////////////////////////////////////
// JaggedDenseDenseElementwiseJaggedOutputOptSearchKernel - Device Kernel
////////////////////////////////////////////////////////////////////////////////
//
// CUDA SOURCE MAPPING:
//   CUDA Kernel:
//     jagged_dense_dense_elementwise_jagged_output_opt_search_kernel_
//   CUDA File: fbgemm_gpu/src/jagged_tensor_ops/common.cuh
//

template <typename index_t>
class JaggedDenseDenseElementwiseJaggedOutputOptSearchKernel {
public:
    JaggedDenseDenseElementwiseJaggedOutputOptSearchKernel(
        at::PackedTensorAccessor32<index_t, 1, RestrictPtrTraits> offsets,
        at::PackedTensorAccessor32<int, 1, RestrictPtrTraits> rows,
        at::PackedTensorAccessor32<int, 1, RestrictPtrTraits> cols,
        int nnz,
        int B,
        sycl::local_accessor<index_t, 1> offsets_sh)
        : offsets_(offsets),
          rows_(rows),
          cols_(cols),
          nnz_(nnz),
          B_(B),
          offsets_sh_(offsets_sh) {}

    void operator()(const sycl::nd_item<1>& item) const {
        index_t* offsets_sh =
            offsets_sh_.template get_multi_ptr<sycl::access::decorated::no>()
                .get();

        for (int64_t i = item.get_local_id(0); i < B_ + 1;
             i += item.get_local_range(0)) {
            offsets_sh[i] = offsets_[i];
        }
        sycl::group_barrier(item.get_group());

        const int row = static_cast<int>(item.get_global_id(0));
        if (row >= nnz_) {
            return;
        }

        // offsets holds B + 1 entries, so searching the full B-wide range
        // [1, B] is what lets this return dense_row == B for a values row that
        // lies past the last offset - the "values has more elements than the
        // last element of offsets" case the generic kernel handles by
        // truncating. CUDA seeds this with B - 1 (common.cuh), stopping at
        // offsets[B - 1] and never examining the final offsets[B] sentinel;
        // such a row then resolves to B - 1, passes the gather kernel's bounds
        // check and wrongly gathers the last dense row instead of zeros.
        // The generic kernel searches offsets.numel() - 1 == B in both sources,
        // so upstream's own two paths disagree on such an input. Deliberate
        // deviation from the CUDA source, which has the same defect.
        int count = B_;
        int first = 1;
        while (count > 0) {
            int idx = first;
            int step = count / 2;
            idx += step;
            if (offsets_sh[idx] <= row) {
                first = ++idx;
                count -= step + 1;
            } else {
                count = step;
            }
        }
        --first;

        const int dense_row = first;
        const int offset = static_cast<int>(offsets_sh[dense_row]);
        const int dense_col = row - offset;
        rows_[row] = dense_row;
        cols_[row] = dense_col;
    }

private:
    at::PackedTensorAccessor32<index_t, 1, RestrictPtrTraits> offsets_;
    // Written by the kernel, so they must stay assignable in operator() const.
    mutable at::PackedTensorAccessor32<int, 1, RestrictPtrTraits> rows_;
    mutable at::PackedTensorAccessor32<int, 1, RestrictPtrTraits> cols_;
    int nnz_;
    int B_;
    sycl::local_accessor<index_t, 1> offsets_sh_;
};

// ============================================================================
// Half vector types and their element-wise appliers
// ============================================================================
//
// CUDA SOURCE MAPPING:
//   CUDA Structs: VecType128 (float4 / half8), VecType64 (float2 / half4),
//                 VecType32 (float / __half2)
//   CUDA Functions: f128, f64, f32, fh
//   CUDA File: fbgemm_gpu/src/jagged_tensor_ops/common.cuh
//
// CUDA unions a half8/half4/__half2 payload with a float4/float2/float
// "transaction type" purely to force a wide, aligned load. sycl::vec already
// provides that: sycl::vec<sycl::half, N> occupies exactly 2*N bytes with
// 2*N-byte alignment, so a single indexed load/store through a reinterpreted
// pointer issues the same 128/64/32-bit transaction.
//

using HalfVec8 = sycl::vec<sycl::half, 8>;
using HalfVec4 = sycl::vec<sycl::half, 4>;
using HalfVec2 = sycl::vec<sycl::half, 2>;

// CUDA: f128 / f64 / f32
template <int N, typename F>
inline void apply_vec(
    sycl::vec<sycl::half, N>& v_out,
    const sycl::vec<sycl::half, N>& x,
    const sycl::vec<sycl::half, N>& y0,
    const sycl::vec<sycl::half, N>& y1,
    F f) {
#pragma unroll
    for (int i = 0; i < N; ++i) {
        v_out[i] = f(
            static_cast<sycl::half>(x[i]),
            static_cast<sycl::half>(y0[i]),
            static_cast<sycl::half>(y1[i]));
    }
}

// CUDA: fh
template <typename F>
inline void apply_scalar(
    sycl::half& v_out,
    const sycl::half& x,
    const sycl::half& y0,
    const sycl::half& y1,
    F f) {
    v_out = f(x, y0, y1);
}

////////////////////////////////////////////////////////////////////////////////
// JaggedDenseDenseElementwiseJaggedOutputOptGatherKernel - Device Kernel
////////////////////////////////////////////////////////////////////////////////
//
// CUDA SOURCE MAPPING:
//   CUDA Kernel:
//     jagged_dense_dense_elementwise_jagged_output_opt_gather_kernel_
//   CUDA File: fbgemm_gpu/src/jagged_tensor_ops/common.cuh
//

template <typename index_t, typename F>
class JaggedDenseDenseElementwiseJaggedOutputOptGatherKernel {
public:
    JaggedDenseDenseElementwiseJaggedOutputOptGatherKernel(
        at::PackedTensorAccessor32<c10::Half, 2, RestrictPtrTraits> values,
        at::PackedTensorAccessor32<c10::Half, 2, RestrictPtrTraits> x_values,
        at::PackedTensorAccessor32<c10::Half, 3, RestrictPtrTraits> y0,
        at::PackedTensorAccessor32<c10::Half, 3, RestrictPtrTraits> y1,
        at::PackedTensorAccessor32<int, 1, RestrictPtrTraits> rows,
        at::PackedTensorAccessor32<int, 1, RestrictPtrTraits> cols,
        const int nnz,
        const int E,
        F f)
        : values_(values),
          x_values_(x_values),
          y0_(y0),
          y1_(y1),
          rows_(rows),
          cols_(cols),
          nnz_(nnz),
          E_(E),
          f_(f) {}

    void operator()(const sycl::nd_item<2>& item) const {
        // CUDA: threadIdx.y + blockIdx.y * blockDim.y
        const int values_row = static_cast<int>(item.get_global_id(0));
        if (values_row >= nnz_) {
            return;
        }
        // CUDA: blockDim.y * gridDim.y
        const int row_stride = static_cast<int>(
            item.get_local_range(0) * item.get_group_range(0));

        // CUDA: threadIdx.x / blockDim.x
        const int tid_begin = static_cast<int>(item.get_local_id(1));
        const int tid_stride = static_cast<int>(item.get_local_range(1));

        for (int real_row = values_row; real_row < nnz_;) {
            const int dense_row = rows_[real_row];
            const int dense_col = cols_[real_row];

            sycl::half* values_ptr =
                reinterpret_cast<sycl::half*>(&values_[real_row][0]);
            const sycl::half* x_ptr =
                reinterpret_cast<const sycl::half*>(&x_values_[real_row][0]);

            if ((dense_col < y0_.size(1)) && (dense_row < y0_.size(0)) &&
                (dense_col < y1_.size(1)) && (dense_row < y1_.size(0)) &&
                (dense_col >= 0) && (dense_row >= 0)) {
                // Formed only once the bounds check has passed. CUDA builds
                // these ahead of the check, which is out-of-range pointer
                // arithmetic for any row the check then rejects - reachable
                // now that the search above can return dense_row == B.
                const sycl::half* y0_ptr = reinterpret_cast<const sycl::half*>(
                    &y0_[dense_row][dense_col][0]);
                const sycl::half* y1_ptr = reinterpret_cast<const sycl::half*>(
                    &y1_[dense_row][dense_col][0]);

                for (int tid = tid_begin; tid < E_ / 8; tid += tid_stride) {
                    HalfVec8 v_out = {};
                    const HalfVec8 v_x =
                        reinterpret_cast<const HalfVec8*>(x_ptr)[tid];
                    const HalfVec8 v_y0 =
                        reinterpret_cast<const HalfVec8*>(y0_ptr)[tid];
                    const HalfVec8 v_y1 =
                        reinterpret_cast<const HalfVec8*>(y1_ptr)[tid];
                    apply_vec<8>(v_out, v_x, v_y0, v_y1, f_);
                    reinterpret_cast<HalfVec8*>(values_ptr)[tid] = v_out;
                }
                // Unreachable tail - see deviation (4).
                for (int tid = tid_begin + (E_ / 8) * 8; tid < E_ / 4;
                     tid += tid_stride) {
                    HalfVec4 v_out = {};
                    const HalfVec4 v_x =
                        reinterpret_cast<const HalfVec4*>(x_ptr)[tid];
                    const HalfVec4 v_y0 =
                        reinterpret_cast<const HalfVec4*>(y0_ptr)[tid];
                    const HalfVec4 v_y1 =
                        reinterpret_cast<const HalfVec4*>(y1_ptr)[tid];
                    apply_vec<4>(v_out, v_x, v_y0, v_y1, f_);
                    reinterpret_cast<HalfVec4*>(values_ptr)[tid] = v_out;
                }
                // Unreachable tail - see deviation (4).
                for (int tid = tid_begin + (E_ / 4) * 4; tid < E_ / 2;
                     tid += tid_stride) {
                    HalfVec2 v_out = {};
                    const HalfVec2 v_x =
                        reinterpret_cast<const HalfVec2*>(x_ptr)[tid];
                    const HalfVec2 v_y0 =
                        reinterpret_cast<const HalfVec2*>(y0_ptr)[tid];
                    const HalfVec2 v_y1 =
                        reinterpret_cast<const HalfVec2*>(y1_ptr)[tid];
                    apply_vec<2>(v_out, v_x, v_y0, v_y1, f_);
                    reinterpret_cast<HalfVec2*>(values_ptr)[tid] = v_out;
                }
                // Unreachable tail - see deviation (4).
                for (int tid = tid_begin + (E_ / 2) * 2; tid < E_;
                     tid += tid_stride) {
                    sycl::half v_out = sycl::half(0.0f);
                    const sycl::half v_x = x_ptr[tid];
                    const sycl::half v_y0 = y0_ptr[tid];
                    const sycl::half v_y1 = y1_ptr[tid];
                    apply_scalar(v_out, v_x, v_y0, v_y1, f_);
                    values_ptr[tid] = v_out;
                }
            } else {
                // Out of bounds: y0/y1 contribute zeros so the result matches
                // the generic kernel's truncated branch f(x, 0, 0). The zero
                // initializers below are load-bearing.
                for (int tid = tid_begin; tid < E_ / 8; tid += tid_stride) {
                    HalfVec8 v_out = {};
                    const HalfVec8 v_y0 = {};
                    const HalfVec8 v_y1 = {};
                    const HalfVec8 v_x =
                        reinterpret_cast<const HalfVec8*>(x_ptr)[tid];
                    apply_vec<8>(v_out, v_x, v_y0, v_y1, f_);
                    reinterpret_cast<HalfVec8*>(values_ptr)[tid] = v_out;
                }
                // Unreachable tail - see deviation (4).
                for (int tid = tid_begin + (E_ / 8) * 8; tid < E_ / 4;
                     tid += tid_stride) {
                    HalfVec4 v_out = {};
                    const HalfVec4 v_y0 = {};
                    const HalfVec4 v_y1 = {};
                    const HalfVec4 v_x =
                        reinterpret_cast<const HalfVec4*>(x_ptr)[tid];
                    apply_vec<4>(v_out, v_x, v_y0, v_y1, f_);
                    reinterpret_cast<HalfVec4*>(values_ptr)[tid] = v_out;
                }
                // Unreachable tail - see deviation (4).
                for (int tid = tid_begin + (E_ / 4) * 4; tid < E_ / 2;
                     tid += tid_stride) {
                    HalfVec2 v_out = {};
                    const HalfVec2 v_y0 = {};
                    const HalfVec2 v_y1 = {};
                    const HalfVec2 v_x =
                        reinterpret_cast<const HalfVec2*>(x_ptr)[tid];
                    apply_vec<2>(v_out, v_x, v_y0, v_y1, f_);
                    reinterpret_cast<HalfVec2*>(values_ptr)[tid] = v_out;
                }
                // Unreachable tail - see deviation (4). CUDA leaves v_y0/v_y1
                // uninitialized here; they are zeroed to keep this defined.
                for (int tid = tid_begin + (E_ / 2) * 2; tid < E_;
                     tid += tid_stride) {
                    sycl::half v_out = sycl::half(0.0f);
                    const sycl::half v_y0 = sycl::half(0.0f);
                    const sycl::half v_y1 = sycl::half(0.0f);
                    const sycl::half v_x = x_ptr[tid];
                    apply_scalar(v_out, v_x, v_y0, v_y1, f_);
                    values_ptr[tid] = v_out;
                }
            }

            // Guarded increment; see the matching comment in
            // JaggedDenseElementwiseDenseOutputKernel.
            if (nnz_ - real_row <= row_stride) {
                break;
            }
            real_row += row_stride;
        }
    }

private:
    // Written by the kernel, so it must stay assignable in operator() const.
    mutable at::PackedTensorAccessor32<c10::Half, 2, RestrictPtrTraits> values_;
    at::PackedTensorAccessor32<c10::Half, 2, RestrictPtrTraits> x_values_;
    at::PackedTensorAccessor32<c10::Half, 3, RestrictPtrTraits> y0_;
    at::PackedTensorAccessor32<c10::Half, 3, RestrictPtrTraits> y1_;
    at::PackedTensorAccessor32<int, 1, RestrictPtrTraits> rows_;
    at::PackedTensorAccessor32<int, 1, RestrictPtrTraits> cols_;
    int nnz_;
    int E_;
    F f_;
};

// ============================================================================
// jagged_dense_dense_elementwise_jagged_output_matches_opt - Host Function
// ============================================================================
//
// CUDA SOURCE MAPPING:
//   CUDA Function: jagged_dense_dense_elementwise_jagged_output_matches_opt
//   CUDA File: fbgemm_gpu/src/jagged_tensor_ops/common.cuh
//

inline bool jagged_dense_dense_elementwise_jagged_output_matches_opt(
    const int& num_jagged_dim,
    const at::Tensor& x_values,
    const std::vector<at::Tensor>& x_offsets,
    const at::Tensor& y_0_reshaped,
    const at::Tensor& y_1_reshaped,
    const at::Tensor& output_values) {
    bool matches = true;
    matches &= (num_jagged_dim == 1);

    // Unit stride embedding dim
    matches &= (x_values.stride(-1) == 1);
    matches &= (output_values.stride(-1) == 1);
    matches &= (y_0_reshaped.stride(-1) == 1);
    matches &= (y_1_reshaped.stride(-1) == 1);

    // Each row is aligned to 128-bit
    matches &= (x_values.stride(-2) % 8 == 0);
    matches &= (output_values.stride(-2) % 8 == 0);
    matches &= (y_0_reshaped.stride(-2) % 8 == 0);
    matches &= (y_1_reshaped.stride(-2) % 8 == 0);

    matches &= (y_0_reshaped.stride(0) % 8 == 0);
    matches &= (y_1_reshaped.stride(0) % 8 == 0);

    // Base addresses aligned to 128-bit
    matches &= (reinterpret_cast<uint64_t>(x_values.data_ptr()) % 16 == 0);
    matches &= (reinterpret_cast<uint64_t>(output_values.data_ptr()) % 16 == 0);
    matches &= (reinterpret_cast<uint64_t>(y_0_reshaped.data_ptr()) % 16 == 0);
    matches &= (reinterpret_cast<uint64_t>(y_1_reshaped.data_ptr()) % 16 == 0);

    // Rows and cols fit into int32_t
    matches &= (y_0_reshaped.size(0) < std::numeric_limits<int>::max());
    matches &= (y_0_reshaped.size(1) < std::numeric_limits<int>::max());

    // The staged offsets must fit in work-group local memory.
    sycl::queue& queue = c10::xpu::getCurrentXPUStream().queue();
    const int64_t max_local_bytes = static_cast<int64_t>(
        queue.get_device().get_info<sycl::info::device::local_mem_size>());
    const int64_t local_kb = max_local_bytes >> 10;
    // Use 2/3 of the available local memory; leave room for L1$.
    const int64_t used_local_kb = round_down(local_kb * 2 / 3, 16);
    TORCH_CHECK(used_local_kb > 0);
    const int64_t used_local_bytes = used_local_kb << 10;

    AT_DISPATCH_INDEX_TYPES(
        x_offsets[0].scalar_type(), "check_local_memory", [&] {
            const auto B = y_0_reshaped.size(0);
            if (static_cast<int64_t>((B + 1) * sizeof(index_t)) >=
                used_local_bytes) {
                matches = false;
            }
        });

    return matches;
}

// ============================================================================
// jagged_dense_elementwise_jagged_output_opt_ - Host Function (fast path)
// ============================================================================
//
// CUDA SOURCE MAPPING:
//   CUDA Function: jagged_dense_elementwise_jagged_output_opt_
//   CUDA File: fbgemm_gpu/src/jagged_tensor_ops/common.cuh
//

template <typename scalar_t, typename F>
void jagged_dense_elementwise_jagged_output_opt_(
    const at::Tensor& x_values,
    const std::vector<at::Tensor>& x_offsets,
    const at::Tensor& y,
    const at::Tensor& output_values,
    F f) {
    check_tensors_on_same_xpu_(x_values, x_offsets, y, output_values);

    const int num_jagged_dim = y.dim() - 2;
    TORCH_CHECK(
        x_offsets.size() == static_cast<size_t>(num_jagged_dim),
        "x_offsets.size(), ",
        x_offsets.size(),
        " != num_jagged_dim, ",
        num_jagged_dim);

    if (y.numel() == 0 || x_values.numel() == 0) {
        return;
    }

    check_jagged_dense_shape_(x_values, x_offsets, y);

    // Canonicalize y to 3D, collapsing jagged dimensions.
    const at::Tensor y_reshaped = y.view({y.size(0), -1, y.size(-1)});

    sycl::queue& queue = c10::xpu::getCurrentXPUStream().queue();

    if (jagged_dense_dense_elementwise_jagged_output_matches_opt(
            num_jagged_dim,
            x_values,
            x_offsets,
            y_reshaped,
            y_reshaped,
            output_values)) {
        AT_DISPATCH_INDEX_TYPES(
            x_offsets[0].scalar_type(), "jagged_indices_fast_path", [&] {
                const auto nnz = output_values.size(0);
                const auto B = y_reshaped.size(0);
                const auto E = y_reshaped.size(2);

                const auto int_options = at::TensorOptions()
                                             .dtype(at::kInt)
                                             .device(x_values.device());
                at::Tensor t_rows_after_bs = at::empty({nnz}, int_options);
                at::Tensor t_cols_after_bs = at::empty({nnz}, int_options);

                // The search kernel indexes offsets as a 1D accessor, so
                // materialize a contiguous copy if needed.
                const at::Tensor offsets_contig = x_offsets[0].contiguous();

                // Binary search. matches_opt() already verified that the
                // (B + 1) staged offsets fit in local memory.
                {
                    constexpr int64_t kThreadsBs =
                        static_cast<int64_t>(kMaxThreads);
                    const int64_t blocks_bs =
                        std::max<int64_t>(1, div_round_up(nnz, kThreadsBs));

                    using SearchKernelT =
                        JaggedDenseDenseElementwiseJaggedOutputOptSearchKernel<
                            index_t>;
                    queue.submit([&](sycl::handler& cgh) {
                        sycl::local_accessor<index_t, 1> offsets_sh(
                            sycl::range<1>(B + 1), cgh);
                        cgh.parallel_for<SearchKernelT>(
                            sycl::nd_range<1>(
                                sycl::range<1>(blocks_bs * kThreadsBs),
                                sycl::range<1>(kThreadsBs)),
                            SearchKernelT(
                                offsets_contig.packed_accessor32<
                                    index_t,
                                    1,
                                    RestrictPtrTraits>(),
                                t_rows_after_bs
                                    .packed_accessor32<int, 1, RestrictPtrTraits>(),
                                t_cols_after_bs
                                    .packed_accessor32<int, 1, RestrictPtrTraits>(),
                                static_cast<int>(nnz),
                                static_cast<int>(B),
                                offsets_sh));
                    });
                }

                // Gather kernel. CUDA: threads (16, 16), blocks (1, ...).
                {
                    constexpr int64_t kThreadsY = 16;
                    constexpr int64_t kThreadsX = 16;
                    int64_t blocks_y =
                        std::max<int64_t>(1, div_round_up(nnz, kThreadsY));
                    if (blocks_y > kJaggedMaxGroups) {
                        blocks_y = kJaggedMaxGroups;
                    }

                    using GatherKernelT =
                        JaggedDenseDenseElementwiseJaggedOutputOptGatherKernel<
                            index_t,
                            F>;
                    queue.submit([&](sycl::handler& cgh) {
                        cgh.parallel_for<GatherKernelT>(
                            sycl::nd_range<2>(
                                sycl::range<2>(blocks_y * kThreadsY, kThreadsX),
                                sycl::range<2>(kThreadsY, kThreadsX)),
                            GatherKernelT(
                                output_values.packed_accessor32<
                                    c10::Half,
                                    2,
                                    RestrictPtrTraits>(),
                                x_values.packed_accessor32<
                                    c10::Half,
                                    2,
                                    RestrictPtrTraits>(),
                                y_reshaped.packed_accessor32<
                                    c10::Half,
                                    3,
                                    RestrictPtrTraits>(),
                                y_reshaped.packed_accessor32<
                                    c10::Half,
                                    3,
                                    RestrictPtrTraits>(),
                                t_rows_after_bs
                                    .packed_accessor32<int, 1, RestrictPtrTraits>(),
                                t_cols_after_bs
                                    .packed_accessor32<int, 1, RestrictPtrTraits>(),
                                static_cast<int>(nnz),
                                static_cast<int>(E),
                                f));
                    });
                }
            });
    } else {
#define INVOKE_KERNEL_WITH_DIM(NUM_JAGGED_DIM) \
    FBGEMM_XPU_JAGGED_OUTPUT_INVOKE_BODY(NUM_JAGGED_DIM)

        FBGEMM_XPU_JAGGED_DISPATCH_DIMS();

#undef INVOKE_KERNEL_WITH_DIM
    }
}

}  // namespace fbgemm_xpu
