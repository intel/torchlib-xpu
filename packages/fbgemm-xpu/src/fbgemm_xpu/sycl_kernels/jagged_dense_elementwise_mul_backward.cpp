/*
 * Copyright (c) Meta Platforms, Inc. and affiliates. All rights reserved.
 * Copyright (c) 2026 Intel Corporation. All Rights Reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */

////////////////////////////////////////////////////////////////////////////////
// SYCL PORT MAPPING TO FBGEMM CUDA SOURCE - JAGGED DENSE ELEMENTWISE MUL
// (BACKWARD)
////////////////////////////////////////////////////////////////////////////////
//
// ORIGINAL CUDA SOURCE:
//   File: fbgemm_gpu/src/jagged_tensor_ops/jagged_dense_elementwise_mul_backward.cu
//   Kernel: fbgemm_gpu::jagged_jagged_elementwise_dense_output_kernel_
//   Functions: fbgemm_gpu::jagged_jagged_elementwise_dense_output_,
//              fbgemm_gpu::jagged_dense_elementwise_mul_backward
//
// See jagged_dense_elementwise_mul_backward.h for the full mapping notes.
//
////////////////////////////////////////////////////////////////////////////////

#include <ATen/Dispatch.h>

#include "fbgemm_utils/utils.h"
#include "jagged_common.h"
#include "jagged_dense_elementwise_mul_backward.h"

namespace fbgemm_xpu {

////////////////////////////////////////////////////////////////////////////////
// JaggedJaggedElementwiseDenseOutputKernel - Device Kernel
////////////////////////////////////////////////////////////////////////////////
//
// CUDA SOURCE MAPPING:
//   CUDA Kernel: jagged_jagged_elementwise_dense_output_kernel_
//   CUDA File: fbgemm_gpu/src/jagged_tensor_ops/jagged_dense_elementwise_mul_backward.cu
//
// DESCRIPTION:
//   output = f(x, y) where x and y are jagged and share x_offsets, and output
//   is dense. padding_value fills the dense positions outside the jagged
//   region; it is never passed to f.
//
////////////////////////////////////////////////////////////////////////////////

template <int NUM_JAGGED_DIM, typename index_t, typename scalar_t, typename F>
class JaggedJaggedElementwiseDenseOutputKernel {
public:
    JaggedJaggedElementwiseDenseOutputKernel(
        at::PackedTensorAccessor32<scalar_t, 2, RestrictPtrTraits> x_values,
        StackArray<const index_t*> x_offsets,
        at::PackedTensorAccessor32<scalar_t, 2, RestrictPtrTraits> y_values,
        at::PackedTensorAccessor32<scalar_t, 3, RestrictPtrTraits> output,
        StackArray<int64_t> jagged_dims,
        F f,
        scalar_t padding_value)
        : x_values_(x_values),
          x_offsets_(x_offsets),
          y_values_(y_values),
          output_(output),
          jagged_dims_(jagged_dims),
          f_(f),
          padding_value_(padding_value) {}

    void operator()(const sycl::nd_item<2>& item) const {
        const int outer_dense_size = output_.size(0);
        const int jagged_folded_size = output_.size(1);
        const int inner_dense_size = output_.size(2);

        // CUDA: blockIdx.x * blockDim.y + threadIdx.y
        const int outer_begin = static_cast<int>(
            item.get_group(0) * item.get_local_range(0) + item.get_local_id(0));
        // CUDA: gridDim.x * blockDim.y
        const int outer_stride = static_cast<int>(
            item.get_group_range(0) * item.get_local_range(0));
        const int total_outer = outer_dense_size * jagged_folded_size;

        // CUDA: threadIdx.x / blockDim.x
        const int inner_begin = static_cast<int>(item.get_local_id(1));
        const int inner_stride = static_cast<int>(item.get_local_range(1));

        // check_shape_and_partition_ sizes the inner extent for element pairs,
        // so the inner loops step over pairs as in
        // JaggedDenseElementwiseDenseOutputKernel; see the comment there for
        // why the bound is the pair count rather than `iidx * 2 + 1`.
        const int inner_pairs = inner_dense_size / 2;
        const bool has_odd_tail = (inner_dense_size & 1) != 0;

        for (int outer = outer_begin; outer < total_outer;) {
            const int oidx = outer / jagged_folded_size;
            const int jidx = outer % jagged_folded_size;

            int offset = oidx;
            const bool is_zero = walk_down_tensor_storage_tree_<NUM_JAGGED_DIM>(
                offset, jidx, jagged_dims_, x_offsets_);

            if (is_zero) {
                int iidx;
                for (iidx = inner_begin; iidx < inner_pairs;
                     iidx += inner_stride) {
                    output_[oidx][jidx][2 * iidx] = padding_value_;
                    output_[oidx][jidx][2 * iidx + 1] = padding_value_;
                }
                if (has_odd_tail && iidx == inner_pairs) {
                    output_[oidx][jidx][2 * iidx] = padding_value_;
                }
            } else {
                int iidx;
                for (iidx = inner_begin; iidx < inner_pairs;
                     iidx += inner_stride) {
                    output_[oidx][jidx][2 * iidx] = f_(
                        x_values_[offset][2 * iidx],
                        y_values_[offset][2 * iidx]);
                    output_[oidx][jidx][2 * iidx + 1] = f_(
                        x_values_[offset][2 * iidx + 1],
                        y_values_[offset][2 * iidx + 1]);
                }
                if (has_odd_tail && iidx == inner_pairs) {
                    output_[oidx][jidx][2 * iidx] = f_(
                        x_values_[offset][2 * iidx],
                        y_values_[offset][2 * iidx]);
                }
            }

            // Guarded increment; see the matching comment in
            // JaggedDenseElementwiseDenseOutputKernel.
            if (total_outer - outer <= outer_stride) {
                break;
            }
            outer += outer_stride;
        }
    }

private:
    at::PackedTensorAccessor32<scalar_t, 2, RestrictPtrTraits> x_values_;
    StackArray<const index_t*> x_offsets_;
    at::PackedTensorAccessor32<scalar_t, 2, RestrictPtrTraits> y_values_;
    // Written by the kernel, so it must stay assignable inside operator() const.
    mutable at::PackedTensorAccessor32<scalar_t, 3, RestrictPtrTraits> output_;
    StackArray<int64_t> jagged_dims_;
    F f_;
    scalar_t padding_value_;
};

////////////////////////////////////////////////////////////////////////////////
// jagged_jagged_elementwise_dense_output_ - Host Function
////////////////////////////////////////////////////////////////////////////////
//
// CUDA SOURCE MAPPING:
//   CUDA Function: jagged_jagged_elementwise_dense_output_
//   CUDA File: fbgemm_gpu/src/jagged_tensor_ops/jagged_dense_elementwise_mul_backward.cu
//
////////////////////////////////////////////////////////////////////////////////

template <typename scalar_t, typename F>
void jagged_jagged_elementwise_dense_output_(
    const at::Tensor& x_values,
    const std::vector<at::Tensor>& x_offsets,
    const at::Tensor& y_values,
    const at::Tensor& output,
    F f,
    const scalar_t padding_value = static_cast<scalar_t>(0)) {
    check_tensors_on_same_xpu_(x_values, x_offsets, y_values, output);

    const int num_jagged_dim = output.dim() - 2;
    TORCH_CHECK(
        x_offsets.size() == static_cast<size_t>(num_jagged_dim),
        "x_offsets.size(), ",
        x_offsets.size(),
        " != num_jagged_dim, ",
        num_jagged_dim);

    if (output.numel() == 0) {
        return;
    }

    const JaggedLaunchConfig cfg =
        check_shape_and_partition_(x_values, x_offsets, output);

    // Canonicalize output to 3D, collapsing jagged dimensions.
    const at::Tensor output_reshaped =
        output.view({output.size(0), -1, output.size(-1)});

    sycl::queue& queue = c10::xpu::getCurrentXPUStream().queue();

#define INVOKE_KERNEL_WITH_DIM(NUM_JAGGED_DIM)                                 \
    {                                                                          \
        StackArray<const index_t*> x_offset_ptrs;                              \
        StackArray<int64_t> x_offset_sizes;                                    \
        const auto x_offsets_contig = collect_offsets_<index_t>(               \
            x_offsets, num_jagged_dim, x_offset_ptrs, x_offset_sizes);         \
                                                                               \
        using KernelT = JaggedJaggedElementwiseDenseOutputKernel<              \
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
                    y_values                                                   \
                        .packed_accessor32<scalar_t, 2, RestrictPtrTraits>(),   \
                    output_reshaped                                            \
                        .packed_accessor32<scalar_t, 3, RestrictPtrTraits>(),   \
                    cfg.jagged_dims,                                           \
                    f,                                                         \
                    padding_value));                                           \
        });                                                                    \
    }

    FBGEMM_XPU_JAGGED_DISPATCH_DIMS();

#undef INVOKE_KERNEL_WITH_DIM
}

////////////////////////////////////////////////////////////////////////////////
// jagged_dense_elementwise_mul_backward_xpu - Host Function
////////////////////////////////////////////////////////////////////////////////
//
// CUDA SOURCE MAPPING:
//   CUDA Function: jagged_dense_elementwise_mul_backward
//   CUDA File: fbgemm_gpu/src/jagged_tensor_ops/jagged_dense_elementwise_mul_backward.cu
//
////////////////////////////////////////////////////////////////////////////////
std::tuple<at::Tensor, at::Tensor> jagged_dense_elementwise_mul_backward_xpu(
    const at::Tensor& grad_output,
    const std::vector<at::Tensor>& x_offsets,
    const at::Tensor& y,
    const at::Tensor& x_values) {
    SYCL_DEVICE_GUARD(grad_output);

    // Deviation (1): the jagged-output launcher writes nothing for an empty y.
    at::Tensor x_values_grad = y.numel() == 0 ? at::zeros_like(grad_output)
                                              : at::empty_like(grad_output);
    at::Tensor y_grad = at::empty_like(y);

    FBGEMM_DISPATCH_FLOATING_TYPES(x_values.scalar_type(), "jagged_scalars", [&] {
        jagged_dense_elementwise_jagged_output_<scalar_t>(
            grad_output, x_offsets, y, x_values_grad, JaggedOpMul());

        jagged_jagged_elementwise_dense_output_<scalar_t>(
            grad_output, x_offsets, x_values, y_grad, JaggedOpMul());
    });

    return {x_values_grad, y_grad};
}

/**
 * Register XPU implementation with PyTorch dispatch system.
 *
 * The schema is already declared by the upstream fbgemm-gpu-cpu package
 * (jagged_tensor_ops_cpu.cpp), which fbgemm_xpu/__init__.py imports before _C
 * loads, so only the XPU dispatch key is registered here.
 */
TORCH_LIBRARY_IMPL(fbgemm, XPU, m) {
    m.impl(
        "jagged_dense_elementwise_mul_backward",
        &jagged_dense_elementwise_mul_backward_xpu);
}

}  // namespace fbgemm_xpu
