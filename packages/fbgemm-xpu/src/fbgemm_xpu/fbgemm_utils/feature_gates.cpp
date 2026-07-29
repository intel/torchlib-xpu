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
#include "fbgemm_utils/function_types.h"

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

bool ev_check_key(const std::string& key) {
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

static bool check_feature_gate_key_impl(
    const std::string& key,
    bool check_env_vars_only [[maybe_unused]]) {
  // Cache feature flags to avoid repeated JK and env var checks
  static std::map<std::string, bool> feature_flags_cache;
  if (const auto search = feature_flags_cache.find(key);
      search != feature_flags_cache.end()) {
    return search->second;
  }

  const auto value = ev_check_key(key);

  feature_flags_cache.insert({key, value});
  return value;
}

DLL_PUBLIC bool check_feature_gate_key(const std::string& key) {
  static const auto no_jk = false;

  return check_feature_gate_key_impl(key, no_jk);
}

DLL_PUBLIC bool is_feature_enabled(const FeatureGateName& feature) {
  return check_feature_gate_key(to_string(feature));
}

DLL_PUBLIC bool is_feature_enabled_from_env(const FeatureGateName& feature) {
  return check_feature_gate_key_impl(
      to_string(feature), /* check_env_vars_only */ true);
}

}  // namespace fbgemm_xpu::config
