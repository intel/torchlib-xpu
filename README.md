[![OpenSSF Scorecard](https://api.securityscorecards.dev/projects/github.com/intel/torchlib-xpu/badge)](https://scorecard.dev/viewer/?uri=github.com/intel/torchlib-xpu)
[![Coverity Scan Build Status](https://scan.coverity.com/projects/intel-torchlib-xpu/badge.svg)](https://scan.coverity.com/projects/intel-torchlib-xpu)

# Intel® XPU Library for PyTorch* Ecosystem Projects

This project contains a set of plugins for PyTorch* ecosystem libraries which enable hardware acceleration on Intel® GPUs thru the `xpu` PyTorch* device backend. The goal of the project is to:

* Facilitate enabling of the Intel® GPUs support across PyTorch* ecosystem projects
* Provide the plugins till the support for Intel® GPUs will be accepted in the respective upstream projects

At the moment project provides plugins for the following frameworks:

* Intel® XPU plugin for [TorchCodec]

## Plugins

### Intel® XPU plugin for TorchCodec

[TorchCodec] is a high-performance Python library designed for media processing (decoding and encoding) using PyTorch* tensors. Intel® XPU plugin for [TorchCodec] enables hardware acceleration for video operations (only decoding at the moment) on Linux. Both [TorchCodec] and Intel® plugin rely on the FFmpeg libraries for their operations which must be pre-installed on the system. Intel® plugin further assumes that FFmpeg is built with the VAAPI support.

[TorchCodec] will automatically load Intel® XPU plugin if it is installed on the system via Python packages entry point mechanism. After importing TorchCodec, pass XPU device to initialize [TorchCodec] decoder or encoder:

```
import torchcodec

decoder = torchcodec.decoders.VideoDecoder(
    "input.mp4", device="xpu:0")
```

## Installation

Pre-built release wheels are available at [PyPI](https://pypi.org/project/torchlib-xpu). Installation requires PyTorch with enabled XPU support which can be fetched from https://download.pytorch.org/whl/xpu:

```
pip install torchlib-xpu \
  --extra-index-url https://download.pytorch.org/whl/xpu
```

## Build from sources

* Install [uv]

* Install oneAPI [2026.0]

* Install FFmpeg development environment with enabled VAAPI hardware acceleration. For example:

  * By installing FFmpeg provided by your Linux distribution. For Ubuntu:

```
apt-get update && apt-get install -y \
    libavcodec-dev \
    libavdevice-dev \
    libavfilter-dev \
    libavformat-dev \
    libavutil-dev \
    libswresample-dev \
    libswscale-dev
```

  * By self-building FFmpeg from sources. The following example provides the minimal configuration required for hardware FFmpeg VAAPI codecs to be functional. For software fallback support, FFmpeg needs to be additionally built with enabled software codecs such as x264, x265, etc.:

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

uv venv && uv pip install -e . \
  --index https://download.pytorch.org/whl/xpu -vv
```

[Getting Started on Intel GPU]: https://docs.pytorch.org/docs/stable/notes/get_start_xpu.html
[TorchCodec]: https://github.com/meta-pytorch/torchcodec
[uv]: https://github.com/astral-sh/uv

[2026.0]: https://www.intel.com/content/www/us/en/developer/articles/tool/pytorch-prerequisites-for-intel-gpu/2-13.html
