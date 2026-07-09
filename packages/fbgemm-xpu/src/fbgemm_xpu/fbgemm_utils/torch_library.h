/*
 * Copyright (c) Meta Platforms, Inc. and affiliates. All rights reserved.
 * Copyright (c) 2026 Intel Corporation. All Rights Reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */

#include <ATen/core/dispatch/Dispatcher.h>

namespace fbgemm_xpu::utils::torch {

inline bool schemaExists(const std::string& qualified_name) {
  return c10::Dispatcher::singleton()
      .findSchema({qualified_name, ""})
      .has_value();
}

} // namespace fbgemm_xpu::utils::torch
