# Intel Plugin for TorchCodec

## Overview

[TorchCodec] is a high-performance Python library designed for media processing (decoding and encoding) using PyTorch* tensors. Intel® XPU plugin for [TorchCodec] enables hardware acceleration for video operations (only decoding at the moment) on Linux. Both [TorchCodec] and Intel® plugin rely on the FFmpeg libraries for their operations which must be pre-installed on the system. Intel® plugin further assumes that FFmpeg is built with the VAAPI support.

To use Intel® XPU plugin for [TorchCodec], load it in the Python script and pass XPU device to initialize [TorchCodec] decoder or encoder:

```
import torchcodec
import torchcodec_xpu

decoder = torchcodec.decoders.VideoDecoder(
    "input.mp4", device="xpu:0")
```

## Supported hardware

All the Intel GPU hardware enabled for XPU PyTorch backend with hardware media decoding capabilities is supported.

## Environment variables

The following environment variables can be used to customize the behavior of Intel Plugin for TorchCodec:

* `USE_SYCL_KERNELS = on|off` (default: `off`) - use SYCL kernels for augmentation such as color space conversion instead of VAAPI interface. If SYCL kernels are requested but can not be used due to hardware limitations, then fallback to VAAPI will be attempted.

## Known limitations

* [Intel® Data Center GPU Max Series][PVC] (Ponte Vecchio, PVC) GPUs are not supported due to missing hardware media engines
* SYCL color space conversion kernel is not supported on [Intel® Arc™ Pro A-Series Graphics][DG2] (Alchemist, DG2) and [Intel® Data Center GPU Flex Series][ATS-M] (Archtic Sound, ATS-M) GPUs as 64-bit floating point operations used in the kernel are not available on these GPUs


[TorchCodec]: https://github.com/meta-pytorch/torchcodec

[ATS-M]: https://www.intel.com/content/www/us/en/ark/products/series/230021/intel-data-center-gpu-flex-series.html
[DG2]: https://www.intel.com/content/www/us/en/ark/products/series/241358/intel-arc-pro-a-series-graphics.html
[PVC]: https://www.intel.com/content/www/us/en/ark/products/series/232874/intel-data-center-gpu-max-series.html
