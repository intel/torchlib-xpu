/*
 * Copyright (c) Meta Platforms, Inc. and affiliates. All rights reserved.
 * Copyright (c) 2026 Intel Corporation. All Rights Reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */

#include <cmath>
#include <cstdint>
#include <tuple>

#include <ATen/core/Tensor.h>
#include <c10/util/Exception.h>
#include <torch/library.h>

namespace fbgemm_xpu {
namespace {

// Re-implementation of fbgemm_gpu's get_info_B_num_bits_from_T (see
// fbgemm_gpu/src/split_embeddings_utils/split_embeddings_utils_cpu.cpp). It is
// pure host integer bit-math with no device code, so we inline it here rather
// than forward-declaring the DLL_PUBLIC symbol from the fbgemm-gpu-cpu wheel.
// Keeping it self-contained avoids a cross-wheel ABI dependency on an internal
// helper and lets the package load _C without promoting fbgemm_gpu's symbols
// via RTLD_GLOBAL.
constexpr int kDefaultInfoNumBits = 32;

int32_t get_num_bits(int32_t n) {
  TORCH_CHECK(n > 0, "Expect n to be positive but got ", n);
  return static_cast<int32_t>(std::floor(std::log2(n) + 1));
}

std::tuple<int32_t, uint32_t> get_info_B_num_bits_from_T(int32_t T, int32_t B) {
  TORCH_CHECK(B > 0, "B must be positive. Got B = ", B);
  TORCH_CHECK(T > 0, "T must be positive. Got T = ", T);
  const int32_t info_T_num_bits = get_num_bits(T);
  const int32_t info_B_num_bits = kDefaultInfoNumBits - info_T_num_bits;
  const uint32_t info_B_mask = (1u << info_B_num_bits) - 1;
  TORCH_CHECK(
      B <= info_B_mask,
      "Not enough infos bits to accommodate T and B. T = ",
      T,
      " takes ",
      info_T_num_bits,
      " and info_B_num_bits is ",
      info_B_num_bits,
      ". Expect max_B = ",
      info_B_mask,
      "but got B ",
      B);
  return {info_B_num_bits, info_B_mask};
}

} // namespace

// Mirrors get_infos_metadata_cpu exactly (note the (T, B) argument order):
//   fbgemm_gpu/src/split_embeddings_utils/split_embeddings_utils_cpu.cpp
// The passed-in tensor is unused and only anchors dispatch to the XPU key.
std::tuple<int64_t, int64_t>
get_infos_metadata_xpu(at::Tensor /*unused*/, int64_t B, int64_t T) {
  return get_info_B_num_bits_from_T(T, B);
}

TORCH_LIBRARY_IMPL(fbgemm, XPU, m) {
  m.impl("get_infos_metadata", &get_infos_metadata_xpu);
}

} // namespace fbgemm_xpu
