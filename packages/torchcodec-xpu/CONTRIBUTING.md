# How to contribute

Open an issue on Github if you've found a problem with the project or have a question.

Submit a PR on Github to propose changes. Before doing so make sure that linter
results are clean and tests are passing.

Before running any checks, make sure to install the project with the test dependencies:

```
uv venv && uv pip install torch~=2.10.0 -e ".[test]" \
  --index https://download.pytorch.org/whl/xpu -vv
```

## How to run linter

```
ruff check
```

## How to run functional tests

At the moment Intel Plugin for TorchCodec uses patched [TorchCodec] tests. To setup:

```
git clone https://github.com/dvrogozh/torchcodec.git && cd torchcodec
git am $TORCHCODEC_XPU_PATH/patches/0001-Add-XPU-support-to-tests.patch
```

The patch is known to apply clean on TorchCodec versions: `v0.10.0`.

Some of the [TorchCodec] tests require FFmpeg with enabled CPU audio and video decoders and encoders. New versions of [TorchCodec] might require more FFmpeg codecs to be enabled. If you self-build FFmpeg, consider to configure all the codec required by [TorchCodec] to reduce number of reported errors on a test run. Note that some of the codecs are GPL licensed. At the moment the following FFmpeg configuration is known to be required to pass [TorchCodec] tests:

```
# Install prerequisites (Ubuntu)
apt-get install \
    libaom-dev \
    libmp3lame-dev \
    libx264-dev \
    libx265-dev \
    libva-dev \
    libvpx-dev

./configure \
    --prefix=$HOME/_install \
    --libdir=$HOME/_install/lib \
    --disable-static \
    --disable-stripping \
    --disable-doc \
    --enable-shared \
    --enable-vaapi \
    --enable-libmp3lame \
    --enable-gpl \
    --enable-libaom \
    --enable-libx264 \
    --enable-libx265 \
    --enable-libvpx
```

TorchCodec tests require some additional packages. Install them as follows:

```
uv pip install torchvision \
  --index https://download.pytorch.org/whl/xpu
```

Finally, execute the tests with:

```
cd torchcodec && pytest test/
```

If tests are flooding the tmpfs, consider to prune pytest directory or use other location:

```
sudo rm -rf /tmp/pytest-of-$(whoami)
# or run as
pytest --basetemp=$HOME/tmp test/
```

# Build and tips

Use the following compatibility table when self-building TorchCodec and its dependencies or using specific versions:

| PyTorch | Torchvision | oneAPI   |
| ------- | ----------- | -------- |
| 2.10    | 0.25        | [2025.3] |
| 2.9     | 0.24        | [2025.2] |
| 2.8     | 0.23        | [2025.1] |

[TorchCodec]: https://github.com/meta-pytorch/torchcodec

[2025.3]: https://www.intel.com/content/www/us/en/developer/articles/tool/pytorch-prerequisites-for-intel-gpu/2-10.html
[2025.2]: https://www.intel.com/content/www/us/en/developer/articles/tool/pytorch-prerequisites-for-intel-gpu/2-9.html
[2025.1]: https://www.intel.com/content/www/us/en/developer/articles/tool/pytorch-prerequisites-for-intel-gpu/2-8.html
