/*
 * Copyright (c) Meta Platforms, Inc. and affiliates. All rights reserved.
 * Copyright (c) 2026 Intel Corporation. All Rights Reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */

#include "asynchronous_complete_cumsum.h"

namespace fbgemm_xpu {

// ============================================================================
// Host Function - XPU Implementation
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
    auto r_out = t_out.slice(0, 1);  // View excluding the first element
    at::cumsum_out(r_out, t_in, 0);   // Compute cumsum into the view
    return t_out;
  }

  // Handle 2D case: cumsum along dimension 1 (columns)
  // input shape [M, N] → output shape [M, N+1]
  at::Tensor t_out = at::zeros({t_in.size(0), t_in.size(1) + 1}, t_in.options());
  auto r_out = t_out.slice(1, 1);   // View excluding the first column
  at::cumsum_out(r_out, t_in, 1);   // Compute cumsum along dim 1 into the view
  return t_out;
}

// The operator registration must happen exactly once. This source file is also
// compiled into the training extension (_C_training), where backward_utils
// calls asynchronous_complete_cumsum_xpu() directly as a C++ helper and does
// not need the dispatcher registration. Guarding it out there prevents a
// duplicate XPU registration for fbgemm::asynchronous_complete_cumsum.
#ifndef FBGEMM_XPU_TRAINING_BUILD
TORCH_LIBRARY_IMPL(fbgemm, XPU, m) {
  m.impl("asynchronous_complete_cumsum", &asynchronous_complete_cumsum_xpu);
}
#endif // FBGEMM_XPU_TRAINING_BUILD

} // namespace fbgemm_xpu
