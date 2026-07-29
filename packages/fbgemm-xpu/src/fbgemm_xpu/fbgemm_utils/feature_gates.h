/*
 * Copyright (c) Meta Platforms, Inc. and affiliates. All rights reserved.
 * Copyright (c) 2026 Intel Corporation. All Rights Reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */

#pragma once

#include <string>

namespace fbgemm_xpu::config {

#define ENUMERATE_ALL_FEATURE_FLAGS       \
  X(TBE_V2)                               \
  X(TBE_ENSEMBLE_ROWWISE_ADAGRAD)         \
  X(TBE_ANNOTATE_KINETO_TRACE)            \
  X(TBE_ROCM_INFERENCE_PACKED_BAGS)       \
  X(TBE_ROCM_HIP_BACKWARD_KERNEL)         \
  X(BOUNDS_CHECK_INDICES_V2)              \
  X(TBE_REPORT_INPUT_PARAMS)              \
  X(TBE_CPU_OUTPUT_DISABLE_PINNED_MEMORY) \
  X(TBE_USE_TUNED_SEGMENT_LENGTHS_CTA_B200)
// X(EXAMPLE_FEATURE_FLAG)

enum class FeatureGateName {
#define X(value) value,
  ENUMERATE_ALL_FEATURE_FLAGS
#undef X
};

std::string to_string(const FeatureGateName& value);

bool check_feature_gate_key(const std::string& key);

bool is_feature_enabled(const FeatureGateName& feature);

bool is_feature_enabled_from_env(const FeatureGateName& feature);


}  // namespace fbgemm_xpu::config
