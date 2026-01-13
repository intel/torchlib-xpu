// Copyright (c) 2026 Intel Corporation. All Rights Reserved.

#pragma once

#ifdef WITH_SYCL_KERNELS

#include <sycl/sycl.hpp>
#include <cstdint>

namespace facebook::torchcodec {

void convertNV12ToRGB(
    sycl::queue& queue,
    const uint8_t* y_plane,
    const uint8_t* uv_plane,
    uint8_t* rgb_output,
    int width,
    int height,
    int stride,
    bool fullrange = 1);

// Anchor function to force kernel registration
void registerColorConversionKernel();

} // namespace facebook::torchcodec

#endif // WITH_SYCL_KERNELS
