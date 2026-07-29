/*
 * Copyright 2026 Intel Corporation
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Portions of this file are derived from FBGEMM
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * SPDX-License-Identifier: BSD-3-Clause
 */

#include "utils.h"

using Tensor = at::Tensor;

namespace fbgemm_xpu {
  DLL_PUBLIC std::tuple<int32_t, uint32_t> adjust_info_B_num_bits(
      int32_t B,
      int32_t T) {
    int32_t info_B_num_bits = kDefaultInfoBNumBits;
    uint32_t info_B_mask = kDefaultInfoBMask;
    uint32_t max_T = kMaxT;
    uint32_t max_B = kMaxB;
    bool invalid_T = T > max_T;
    bool invalid_B = B > max_B;

    TORCH_CHECK(
        !(invalid_T && invalid_B),
        "Not enough infos bits to accommodate T and B. Default num bits = ",
        kDefaultInfoNumBits);

    if (invalid_T) {
      // Reduce info_B_num_bits
      while (invalid_T && !invalid_B && info_B_num_bits > 0) {
        info_B_num_bits--;
        max_T = ((max_T + 1) << 1) - 1;
        max_B = ((max_B + 1) >> 1) - 1;
        invalid_T = T > max_T;
        invalid_B = B > max_B;
      }
    } else if (invalid_B) {
      // Increase info_B_num_bits
      while (!invalid_T && invalid_B && info_B_num_bits < kDefaultInfoNumBits) {
        info_B_num_bits++;
        max_T = ((max_T + 1) >> 1) - 1;
        max_B = ((max_B + 1) << 1) - 1;
        invalid_T = T > max_T;
        invalid_B = B > max_B;
      }
    }

    TORCH_CHECK(
        !invalid_T && !invalid_B,
        "Not enough infos bits to accommodate T and B. Default num bits = ",
        kDefaultInfoNumBits);

    // Recompute info_B_mask using new info_B_num_bits
    info_B_mask = (1u << info_B_num_bits) - 1;

    return {info_B_num_bits, info_B_mask};
  }
};