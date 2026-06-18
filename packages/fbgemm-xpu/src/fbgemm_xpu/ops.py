# Copyright (c) Meta Platforms, Inc. and affiliates. All rights reserved.
# Copyright (c) 2026 Intel Corporation. All Rights Reserved.
# SPDX-License-Identifier: BSD-3-Clause

# Python wrapper functions for all custom operators under the fbgemm namespace
# This module provides user-friendly interfaces to the C++ operators

import torch
from torch import Tensor

__all__ = [
    "invert_permute",
]

def invert_permute(permute: Tensor) -> Tensor:
    """Computes the inverse of a permutation tensor."""
    return torch.ops.fbgemm.invert_permute.default(permute)
