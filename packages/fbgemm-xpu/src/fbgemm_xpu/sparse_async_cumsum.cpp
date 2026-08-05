/*
 * Copyright (c) Meta Platforms, Inc. and affiliates. All rights reserved.
 * Copyright (c) 2026 Intel Corporation. All Rights Reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */

#include "sparse_async_cumsum.h"

namespace fbgemm_xpu {
// TODO: Implement cumulative sum operators as native SYCL kernel for better performance.
// Currently using PyTorch's cumulative sum for testing purposes.
// CCCL's DeviceScan::ExclusiveSum uses a sophisticated
// tile-based "decoupled look-back" algorithm (see: https://research.nvidia.com/publication/single-pass-parallel-prefix-scan-decoupled-look-back)
// Porting this to SYCL would require:
// 1. Understanding the tile-state coordination mechanism used by CCCL's lookback/lookahead algorithms
// 2. Profiling to validate the implementation meets or exceeds PyTorch performance
// This optimization is deferred until cumulative sum becomes a measured bottleneck in DLRM workloads.

// ============================================================================
// Host Functions - XPU Implementation
// ============================================================================

at::Tensor asynchronous_complete_cumsum_xpu(const at::Tensor& t_in) {
  // Input validation
  TORCH_CHECK(t_in.is_contiguous(), "Input tensor must be contiguous");
  TORCH_CHECK(
      t_in.dtype() == at::kInt || t_in.dtype() == at::kLong,
      "Input tensor must have dtype int32 or int64");
  TORCH_CHECK(
      t_in.dim() == 1 || t_in.dim() == 2,
      "Input tensor must be 1D or 2D");

  // Handle 1D case: input [a, b, c] → output [0, a, a+b, a+b+c]
  if (t_in.dim() == 1) {
    at::Tensor t_out = at::zeros({t_in.numel() + 1}, t_in.options());
    if (t_in.numel() == 0) {
      return t_out;
    }
    auto r_out = t_out.slice(0, 1);  // View excluding the first element
    at::cumsum_out(r_out, t_in, 0);   // Compute cumsum into the view
    return t_out;
  }

  // Handle 2D case: cumsum along dimension 1 (columns)
  // input shape [M, N] → output shape [M, N+1]
  at::Tensor t_out = at::zeros({t_in.size(0), t_in.size(1) + 1}, t_in.options());
  if (t_in.numel() == 0) {
    return t_out;
  }
  auto r_out = t_out.slice(1, 1);   // View excluding the first column
  at::cumsum_out(r_out, t_in, 1);   // Compute cumsum along dim 1 into the view
  return t_out;
}

at::Tensor asynchronous_exclusive_cumsum_xpu(const at::Tensor& t_in) {
  // Input validation
  TORCH_CHECK(t_in.is_contiguous(), "Input tensor must be contiguous");
  TORCH_CHECK(
      t_in.dtype() == at::kInt || t_in.dtype() == at::kLong,
      "Input tensor must have dtype int32 or int64");

  // Handle empty input
  if (t_in.numel() == 0) {
    return at::empty_like(t_in);
  }

  // Exclusive cumsum: input [a, b, c] → output [0, a, a+b]
  // This is computed as inclusive cumsum shifted right by one position
  at::Tensor t_out = at::empty_like(t_in);
  
  // Set first element to 0
  t_out[0] = 0;
  
  // If there's more than one element, compute inclusive cumsum of input[:-1]
  // and place it in output[1:]
  if (t_in.numel() > 1) {
    auto in_slice = t_in.slice(0, 0, t_in.numel() - 1);  // input[:-1]
    auto out_slice = t_out.slice(0, 1);                   // output[1:]
    at::cumsum_out(out_slice, in_slice, 0);
  }
  
  return t_out;
}

at::Tensor asynchronous_inclusive_cumsum_xpu(const at::Tensor& t_in) {
  // Input validation
  TORCH_CHECK(t_in.is_contiguous(), "Input tensor must be contiguous");
  TORCH_CHECK(
      t_in.dtype() == at::kInt || t_in.dtype() == at::kLong,
      "Input tensor must have dtype int32 or int64");

  // Handle empty input
  if (t_in.numel() == 0) {
    return at::empty_like(t_in);
  }

  // Inclusive cumsum: input [a, b, c] → output [a, a+b, a+b+c]
  // Use cumsum_out to preserve dtype (at::cumsum may promote int32 to int64)
  at::Tensor t_out = at::empty_like(t_in);
  at::cumsum_out(t_out, t_in, 0);
  return t_out;
}

TORCH_LIBRARY_IMPL(fbgemm, XPU, m) {
  m.impl("asynchronous_complete_cumsum", &fbgemm_xpu::asynchronous_complete_cumsum_xpu);
  m.impl("asynchronous_exclusive_cumsum", &fbgemm_xpu::asynchronous_exclusive_cumsum_xpu);
  m.impl("asynchronous_inclusive_cumsum", &fbgemm_xpu::asynchronous_inclusive_cumsum_xpu);
}

} // namespace fbgemm_xpu
