# Copyright (c) 2026 Intel Corporation. All Rights Reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""Checks that FBGEMM selects ComputeDevice.XPU on this machine."""

import torch
from fbgemm_gpu.split_table_batched_embeddings_ops_training import (
    get_available_compute_device,
)
from fbgemm_gpu.tbe.config import ComputeDevice

assert torch.xpu.is_available(), "XPU (Intel GPU) is not available"
assert ComputeDevice.XPU == 3, list(ComputeDevice)
assert ComputeDevice.get_available() == ComputeDevice.XPU, ComputeDevice.get_available()
assert get_available_compute_device() == ComputeDevice.XPU, get_available_compute_device()
print("ComputeDevice:", list(ComputeDevice))
