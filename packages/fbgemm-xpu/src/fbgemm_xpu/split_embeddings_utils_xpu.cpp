/*
 * Copyright (c) Meta Platforms, Inc. and affiliates. All rights reserved.
 * Copyright (c) 2026 Intel Corporation. All Rights Reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */

#include <cstdint>
#include <tuple>

#include <ATen/core/Tensor.h>
#include <torch/library.h>

// Reuse case (no device kernel): get_infos_metadata is pure integer bit-math.
// The shared host helper get_info_B_num_bits_from_T is defined DLL_PUBLIC at
// global scope in the fbgemm-gpu-cpu wheel
// (fbgemm_gpu/src/split_embeddings_utils/split_embeddings_utils_cpu.cpp) and
// declared in fbgemm_gpu/include/fbgemm_gpu/split_embeddings_utils.h. We
// forward-declare it here rather than depend on FBGEMM headers; the symbol is
// resolved at load time because fbgemm_xpu/__init__.py imports fbgemm_gpu
// before loading this extension's _C.
std::tuple<int32_t, uint32_t> get_info_B_num_bits_from_T(int32_t T, int32_t B);

namespace fbgemm_xpu {

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
