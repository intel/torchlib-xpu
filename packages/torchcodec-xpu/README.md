# Intel Plugin for TorchCodec

[TorchCodec] is a high-performance Python library designed for media processing (decoding and encoding) using PyTorch* tensors. Intel® XPU plugin for [TorchCodec] enables hardware acceleration for video operations (only decoding at the moment) on Linux. Both TorchCodec and Intel® plugin rely on the FFmpeg libraries for their operations which must be pre-installed on the system. Intel® plugin further assumes that FFmpeg is built with the VAAPI support.

To use Intel® XPU plugin for [TorchCodec], load it in the Python script and pass XPU device to initialize TorchCodec decoder or encoder:

```
import torchcodec
import torchcodec_xpu

decoder = torchcodec.decoders.VideoDecoder(
    "input.mp4", device="xpu:0")
```

[TorchCodec]: https://github.com/meta-pytorch/torchcodec
