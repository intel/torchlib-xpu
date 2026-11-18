/*
 * Copyright (c) Meta Platforms, Inc. and affiliates. All rights reserved.
 * Copyright (c) 2026 Intel Corporation. All Rights Reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */

////////////////////////////////////////////////////////////////////////////////
// SYCL PORT MAPPING TO FBGEMM CUDA SOURCE - JAGGED TO PADDED DENSE (FORWARD)
////////////////////////////////////////////////////////////////////////////////
//
// This file contains the SYCL/XPU host implementation of the
// jagged_to_padded_dense_forward operator, ported from FBGEMM CUDA
//
// ORIGINAL CUDA SOURCE:
//   File: fbgemm_gpu/src/jagged_tensor_ops/jagged_to_padded_dense_forward.cu
//
// KERNEL MAPPING:
//   None of its own. The operator is expressed entirely through the shared
//   dense-output launcher in jagged_common.h:
//
//     JaggedDenseElementwiseDenseOutputKernel<...>
//       → jagged_dense_elementwise_dense_output_kernel_ (CUDA)
//       CUDA File: fbgemm_gpu/src/jagged_tensor_ops/common.cuh
//
// HOST FUNCTION MAPPING:
//   jagged_to_padded_dense_forward_xpu (SYCL)
//     → jagged_to_padded_dense_forward (CUDA)
//     CUDA File: fbgemm_gpu/src/jagged_tensor_ops/jagged_to_padded_dense_forward.cu
//
////////////////////////////////////////////////////////////////////////////////

#pragma once

#include <vector>

#include <ATen/ATen.h>
#include <c10/core/SymIntArrayRef.h>

namespace fbgemm_xpu {

////////////////////////////////////////////////////////////////////////////////
// jagged_to_padded_dense_forward_xpu - Host Function
////////////////////////////////////////////////////////////////////////////////
//
// CUDA SOURCE MAPPING:
//   CUDA Function: jagged_to_padded_dense_forward
//   CUDA File: fbgemm_gpu/src/jagged_tensor_ops/jagged_to_padded_dense_forward.cu
//
////////////////////////////////////////////////////////////////////////////////

at::Tensor jagged_to_padded_dense_forward_xpu(
    const at::Tensor& values,
    const std::vector<at::Tensor>& offsets,
    c10::SymIntArrayRef max_lengths,
    const double padding_value);

}  // namespace fbgemm_xpu
