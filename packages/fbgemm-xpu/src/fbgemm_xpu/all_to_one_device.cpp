/*
 * Copyright (c) Meta Platforms, Inc. and affiliates. All rights reserved.
 * Copyright (c) 2026 Intel Corporation. All Rights Reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */

#include <ATen/core/Tensor.h>
#include <ATen/xpu/PeerToPeerAccess.h>
#include <c10/core/DeviceGuard.h>
#include <torch/library.h>

#include <vector>

#include "fbgemm_utils/tensor_utils.h"

namespace fbgemm_xpu {
namespace {

// XPU counterpart of fbgemm_gpu::all_to_one_device (see fbgemm_gpu/src/
// merge_pooled_embedding_ops/merge_pooled_embedding_ops_gpu.cpp).
// It keeps the CUDA contract: the target needs a device index, tensors already
// on the target are returned as they are, and copies are contiguous. CUDA
// copies the rest through its own peer-to-peer path and requires peer access
// between all devices; here Tensor.to lets PyTorch's XPU cross-device copy
// order the source and target streams, and devices without peer access fall
// back to a synchronous copy.
std::vector<at::Tensor> all_to_one_device_xpu(
    std::vector<at::Tensor> input_tensors,
    at::Device target_device) {
  TORCH_CHECK(
      target_device.is_xpu(), "all_to_one_device: target_device must be XPU");
  TORCH_CHECK(
      target_device.has_index(),
      "target_device.index() is -1. Please pass target_device with device "
      "index, e.g., torch.device(\"xpu:0\")");
  for (const auto& tensor : input_tensors) {
    TENSOR_ON_SYCL_XPU(tensor);
  }

  const c10::DeviceGuard guard(target_device);
  std::vector<at::Tensor> output_tensors;
  output_tensors.reserve(input_tensors.size());
  for (const auto& tensor : input_tensors) {
    if (tensor.device() == target_device) {
      output_tensors.push_back(tensor);
      continue;
    }
    // Without peer access PyTorch stages the copy through pageable host memory
    // and does not wait for the device-to-host half when non_blocking is set.
    const bool non_blocking = at::xpu::get_p2p_access(
        tensor.device().index(), target_device.index());
    output_tensors.push_back(tensor.to(
        target_device,
        tensor.scalar_type(),
        non_blocking,
        /*copy=*/false,
        at::MemoryFormat::Contiguous));
  }
  return output_tensors;
}

} // namespace

TORCH_LIBRARY_IMPL(fbgemm, XPU, m) {
  m.impl("all_to_one_device", TORCH_FN(all_to_one_device_xpu));
}

} // namespace fbgemm_xpu
