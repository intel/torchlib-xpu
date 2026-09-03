/*
 * Copyright (c) Meta Platforms, Inc. and affiliates. All rights reserved.
 * Copyright (c) 2026 Intel Corporation. All Rights Reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */

#include "utils.h"

using Tensor = at::Tensor;

namespace fbgemm_xpu {
  std::tuple<int32_t, uint32_t> adjust_info_B_num_bits(
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
