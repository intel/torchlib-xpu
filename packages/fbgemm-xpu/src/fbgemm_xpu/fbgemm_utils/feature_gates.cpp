/*
 * Copyright (c) Meta Platforms, Inc. and affiliates. All rights reserved.
 * Copyright (c) 2026 Intel Corporation. All Rights Reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */

#include <cstdlib>
#include <map>
#include <stdexcept>
#include <string>

#include "fbgemm_utils/feature_gates.h"

namespace fbgemm_xpu::config {

std::string to_string(const FeatureGateName& value) {
  switch (value) {
#define X(value)               \
  case FeatureGateName::value: \
    return #value;
    ENUMERATE_ALL_FEATURE_FLAGS
#undef X
  }
  return "UNKNOWN";
}

static bool env_check_key(const std::string& key) {
  const auto env_var = "FBGEMM_" + key;

  const auto value = std::getenv(env_var.c_str());
  if (!value) {
    return false;
  }

  try {
    return std::stoi(value) == 1;
  } catch (const std::invalid_argument&) {
    return false;
  }
}

bool check_feature_gate_key(const std::string& key) {
  // Cache feature flags to avoid repeated environment lookups.
  static std::map<std::string, bool> feature_flags_cache;
  if (const auto search = feature_flags_cache.find(key);
      search != feature_flags_cache.end()) {
    return search->second;
  }

  const auto value = env_check_key(key);

  feature_flags_cache.insert({key, value});
  return value;
}

bool is_feature_enabled(const FeatureGateName& feature) {
  return check_feature_gate_key(to_string(feature));
}

}  // namespace fbgemm_xpu::config
