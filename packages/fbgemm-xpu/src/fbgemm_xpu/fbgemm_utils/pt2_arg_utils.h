/*
 * Copyright (c) Meta Platforms, Inc. and affiliates. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */

#pragma once

namespace fbgemm_xpu {

enum ArgIndex_aux_tensor {
    IDX_B_OFFSETS = 0,
    IDX_VBE_OUTPUT_OFFSETS_FEATURE_RANK = 1,
    IDX_VBE_B_OFFSETS_RANK_PER_FEATURE = 2,
    IDX_LXU_CACHE_LOCATIONS = 3,
    IDX_UVM_CACHE_STATS = 4,
    IDX_PREV_ITER_DEV = 5,
    IDX_VBE_OUTPUT_OFFSETS = 6,
    AUX_TENSOR_SIZE = 7
};
static_assert(AUX_TENSOR_SIZE == IDX_VBE_OUTPUT_OFFSETS + 1);

enum ArgIndex_aux_bool {
    IDX_IS_EXPERIMENTAL_TBE = 0,
    IDX_USE_UNIQ_CACHE_LOCATIONS_BWD = 1,
    IDX_USE_HOMOGENEOUS_PLACEMENTS = 2,
    IDX_APPLY_GLOBAL_WEIGHT_DECAY = 3,
    IDX_GRADIENT_CLIPPING = 4,
    IDX_STOCHASTIC_ROUNDING = 5,
    IDX_MIXED_D = 6,
    AUX_BOOL_SIZE = 7
};
static_assert(AUX_BOOL_SIZE == IDX_MIXED_D + 1);

enum ArgIndex_aux_int {
    IDX_ITER = 0,
    IDX_INFO_B_NUM_BITS = 1,
    IDX_INFO_B_MASK = 2,
    AUX_INT_SIZE = 3
};
static_assert(AUX_INT_SIZE == IDX_INFO_B_MASK + 1);

enum ArgIndex_aux_float {
    IDX_GWD_LOWER_BOUND = 0,
    IDX_MAX_GRADIENT = 1,
    AUX_FLOAT_SIZE = 2
};
static_assert(AUX_FLOAT_SIZE == IDX_MAX_GRADIENT + 1);

}  // namespace fbgemm_xpu
