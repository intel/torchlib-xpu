# Intel® XPU Library for PyTorch* Ecosystem Projects

This project contains a set of plugins for PyTorch* ecosystem libraries which enable hardware acceleration on Intel® GPUs thru the `xpu` PyTorch* device backend. The goal of the project is to:

* Facilitate enabling of the Intel® GPUs support across PyTorch* ecosystem projects
* Provide the plugins till the support for Intel® GPUs will be accepted in the respective upstream projects

At the moment project provides plugins for the following frameworks:

* Intel® XPU plugin for [TorchCodec]

## Plugins

### Intel® XPU plugin for TorchCodec

[TorchCodec] is a high-performance Python library designed for media processing (decoding and encoding) using PyTorch* tensors. Intel® XPU plugin for [TorchCodec] enables hardware acceleration for video operations (only decoding at the moment) on Linux. Both [TorchCodec] and Intel® plugin rely on the FFmpeg libraries for their operations which must be pre-installed on the system. Intel® plugin further assumes that FFmpeg is built with the VAAPI support.

To use Intel® XPU plugin for [TorchCodec], load it in the Python script and pass XPU device to initialize [TorchCodec] decoder or encoder:

```
import torchcodec
import torchcodec_xpu

decoder = torchcodec.decoders.VideoDecoder(
    "input.mp4", device="xpu:0")
```

## Build from sources

* Install [uv]

* Install oneAPI [2025.3]

* Install FFmpeg with enabled VAAPI hardware acceleration. For example:

```
git clone https://git.ffmpeg.org/ffmpeg.git && cd ffmpeg
./configure \
  --prefix=$HOME/_install \
  --libdir=$HOME/_install/lib \
  --disable-static \
  --disable-stripping \
  --disable-doc \
  --enable-shared \
  --enable-vaapi
make -j$(nproc) && make install

export PKG_CONFIG_PATH=$HOME/_install/lib/pkgconfig
export LD_LIBRARY_PATH=$HOME/_install/lib:$LD_LIBRARY_PATH
```

* Build and install plugins supplied by Intel® XPU Library for PyTorch* Ecosystem Projects:

```
git clone https://github.com/intel/torchlib-xpu.git && cd torchlib-xpu

uv venv && uv pip install torch~=2.10.0 -e . \
  --index https://download.pytorch.org/whl/xpu -vv
```

[Getting Started on Intel GPU]: https://docs.pytorch.org/docs/stable/notes/get_start_xpu.html
[TorchCodec]: https://github.com/meta-pytorch/torchcodec
[uv]: https://github.com/astral-sh/uv

[2025.3]: https://www.intel.com/content/www/us/en/developer/articles/tool/pytorch-prerequisites-for-intel-gpu/2-10.html
