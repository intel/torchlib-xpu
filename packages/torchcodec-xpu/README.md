# Intel Plugin for TorchCodec

This repo contains a prototype of the Intel Plugin for [TorchCodec].

# Prerequisites

To run:
* System with Intel Xe GPU capable to:
  * Run PyTorch* XPU backend
  * Run HW accelerated video decoding
* Linux operating system
* FFmpeg version 6 or later with enabled VAAPI hardware acceleration
* PyTorch* with enabled XPU backend
* TorchCodec version of v0.10.x series

Additionally, to build:
* CMake 3.18 or later
* Intel oneAPI matching the version used to build PyTorch

# How to build

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
```

* Build and install Intel plugin for TorchCodec:

```
git clone https://github.com/dvrogozh/torchcodec-xpu.git
cd torchcodec-xpu

# If ffmpeg was installed per above example
export PKG_CONFIG_PATH=$HOME/_install/lib/pkgconfig
export LD_LIBRARY_PATH=$HOME/_install/lib:$LD_LIBRARY_PATH

uv venv && uv pip install torch~=2.10.0 -e . \
  --index https://download.pytorch.org/whl/xpu -vv
```

# How to use

Import the project in your Python script to register Intel device
interface for the TorchCodec:

```
import torchcodec
import torchcodec_xpu
```

Before executing, make sure to export environment variables to point to FFmpeg installation:

```
export PATH=$HOME/_install/bin:$PATH
export LD_LIBRARY_PATH=$HOME/_install/lib:$LD_LIBRARY_PATH
```

[Getting Started on Intel GPU]: https://docs.pytorch.org/docs/stable/notes/get_start_xpu.html
[TorchCodec]: https://github.com/meta-pytorch/torchcodec

[2025.3]: https://www.intel.com/content/www/us/en/developer/articles/tool/pytorch-prerequisites-for-intel-gpu/2-10.html
