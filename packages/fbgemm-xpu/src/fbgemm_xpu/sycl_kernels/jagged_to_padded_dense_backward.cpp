/*
 * Copyright (c) Meta Platforms, Inc. and affiliates. All rights reserved.
 * Copyright (c) 2026 Intel Corporation. All Rights Reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */

////////////////////////////////////////////////////////////////////////////////
// SYCL PORT MAPPING TO FBGEMM CUDA SOURCE - JAGGED TO PADDED DENSE (BACKWARD)
////////////////////////////////////////////////////////////////////////////////
//
// ORIGINAL CUDA SOURCE:
//   File: fbgemm_gpu/src/jagged_tensor_ops/jagged_to_padded_dense_backward.cu
//   Function: fbgemm_gpu::jagged_to_padded_dense_backward
//
// See jagged_to_padded_dense_backward.h for the full mapping notes.
//
////////////////////////////////////////////////////////////////////////////////

#include <ATen/Dispatch.h>

#include "fbgemm_utils/utils.h"
#include "jagged_common.h"
#include "jagged_to_padded_dense_backward.h"

namespace fbgemm_xpu {

////////////////////////////////////////////////////////////////////////////////
// jagged_to_padded_dense_backward_xpu - Host Function
////////////////////////////////////////////////////////////////////////////////
//
// CUDA SOURCE MAPPING:
//   CUDA Function: jagged_to_padded_dense_backward
//   CUDA File: fbgemm_gpu/src/jagged_tensor_ops/jagged_to_padded_dense_backward.cu
//
////////////////////////////////////////////////////////////////////////////////
at::Tensor jagged_to_padded_dense_backward_xpu(
    const at::Tensor& grad_output,
    const std::vector<at::Tensor>& offsets,
    at::SymInt total_L) {
    auto grad_padded_values = grad_output;
    SYCL_DEVICE_GUARD(grad_padded_values);

    // Canonicalize padded_values by unsqueezing the last dim if the inner dense
    // dimension is 1 and folded.
    const bool D_folded = grad_padded_values.dim() == offsets.size() + 1;
    at::Tensor grad_padded_values_view =
        D_folded ? grad_padded_values.unsqueeze(-1) : grad_padded_values;
    const int32_t D = grad_padded_values_view.size(-1);

    // Initialize with zeros. The kernel also writes zeros to the portion
    // truncated in forward, but it writes nothing when grad_output is empty.
    auto grad_values =
        at::zeros_symint({total_L, D}, grad_padded_values.options());

    FBGEMM_DISPATCH_FLOATING_TYPES(
        grad_padded_values.scalar_type(),
        "jagged_to_dense_backward_kernel",
        [&] {
            jagged_dense_elementwise_jagged_output_<scalar_t>(
                // Passed as the jagged operand only to supply shape;
                // JaggedOpCopyY ignores it, exactly as the CUDA lambda ignores
                // its first argument.
                grad_values,
                offsets,
                grad_padded_values_view,
                grad_values,
                JaggedOpCopyY());
        });

    return D_folded ? grad_values.squeeze(-1) : grad_values;
}

TORCH_LIBRARY_IMPL(fbgemm, XPU, m) {
    m.impl(
        "jagged_to_padded_dense_backward",
        &jagged_to_padded_dense_backward_xpu);
}

}  // namespace fbgemm_xpu
