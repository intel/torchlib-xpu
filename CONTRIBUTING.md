# Contributing

## License

The project is licensed under the terms in [LICENSE](./LICENSE). By contributing to the project, you agree to the license and copyright terms therein and release your contribution under these terms.

Please use the sign-off line at the end of the patch. Your signature certifies that you wrote the patch or otherwise have the right to pass it on as an open-source patch following http://developercertificate.org/ certification. The sign-off line must look like the following:

```
Signed-off-by: Joe Smith <joe.smith@email.com>
```

Use your real name (sorry, no pseudonyms or anonymous contributions). If you set your `user.name` and `user.email` git configs, you can sign your commit automatically with `git commit -s`.

## How to contribute

Open an issue on Github if you've found a problem with the project or have a question.

Submit a PR on Github to propose changes. Before doing so make sure that linter results are clean and tests are passing.

Before running any checks, make sure to install the project with the test dependencies:

```
uv venv && uv pip install -e ".[test]" \
  --index https://download.pytorch.org/whl/xpu -vv
```

## How to run linter

```
ruff check
```

## How to test Intel® plugin for TorchCodec

At the moment Intel® Plugin for [TorchCodec] uses patched TorchCodec tests. To setup the testing environment, run:

```
export TORCHCODEC_XPU_PATH=$TORCHLIB_XPU_PATH/packages/torchcodec-xpu/
git clone https://github.com/meta-pytorch/torchcodec.git && cd torchcodec
git am $TORCHCODEC_XPU_PATH/patches/0001-Add-XPU-support-to-tests.patch
```

The patch file can be reviewed [here](packages/torchcodec-xpu/patches/0001-Add-XPU-support-to-tests.patch).

TorchCodec tests require some additional packages. Install them as follows:

```
uv pip install torchvision \
  --index https://download.pytorch.org/whl/xpu
```

Execute the tests with:

```
cd torchcodec && pytest test/
```

If tests are flooding the tmpfs, consider to prune pytest directory or use other location:

```
sudo rm -rf /tmp/pytest-of-$(whoami)
# or run as
pytest --basetemp=$HOME/tmp test/
```

Some of the TorchCodec tests require FFmpeg with enabled CPU audio and video decoders and encoders. New versions of TorchCodec might require more FFmpeg codecs to be enabled. If you self-build FFmpeg, consider to configure all codecs required by TorchCodec to reduce number of reported errors on a test run. Note that some of the codecs are GPL licensed. At the moment the following FFmpeg configuration is known to be required to pass TorchCodec tests:

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

## How to test Intel® plugin for FBGEMM

Intel® Plugin for [FBGEMM] is tested with its own tests and with patched upstream FBGEMM tests. Both require `fbgemm-xpu` installed with the test dependencies:

```
uv pip install -e "packages/fbgemm-xpu[test]" \
  --index https://download.pytorch.org/whl/xpu
```

The plugin tests cover XPU/SYCL specific cases that upstream does not, such as launch-grid limits, device validation and strided fast-path inputs. Run them with:

```
pytest -rsf $TORCHLIB_XPU_PATH/packages/fbgemm-xpu/tests/
```

The launch-grid tests in `test_jagged_large_grid.py` and `test_jagged_loop_bounds_overflow.py` need 3-18 GiB of device memory and are skipped when the device does not have it free. They are the only guard against an int32 launch-grid overflow silently dropping work, so run them on a large-memory device before changing jagged kernel launch geometry. CI sets `FBGEMM_XPU_TEST_SHARED_GPU=1`, which also skips any of them needing more than half of the device's memory, because two test jobs share one GPU there; leave it unset for manual runs.

The upstream FBGEMM tests cover shared operator behavior and parity with the CPU implementation. To setup the testing environment, apply the tests patch to the pinned FBGEMM release:

```
export FBGEMM_XPU_PATH=$TORCHLIB_XPU_PATH/packages/fbgemm-xpu/
git clone --branch v1.9.0 https://github.com/pytorch/FBGEMM.git && cd FBGEMM
git am $FBGEMM_XPU_PATH/patches/0001-Add-XPU-support-to-fbgemm-tests.patch
```

The patch file can be reviewed [here](packages/fbgemm-xpu/patches/0001-Add-XPU-support-to-fbgemm-tests.patch). It changes only `fbgemm_gpu/test/`, and CI enforces this.

Only the operators implemented by the plugin are covered, so run the test files individually instead of the whole `fbgemm_gpu/test/` directory:

```
cd FBGEMM
pytest -rsf fbgemm_gpu/test/jagged/expand_into_jagged_permute_test.py
pytest -rsf fbgemm_gpu/test/jagged/jagged_index_select_2d_test.py
pytest -rsf fbgemm_gpu/test/jagged/dense_to_jagged_test.py
pytest -rsf fbgemm_gpu/test/jagged/jagged_to_padded_dense_test.py
pytest -rsf fbgemm_gpu/test/jagged/2d_to_dense_test.py
pytest -rsf fbgemm_gpu/test/jagged/1d_to_dense_test.py
pytest -rsf fbgemm_gpu/test/jagged/elementwise_binary_test.py
pytest -rsf fbgemm_gpu/test/sparse/block_bucketize_test.py
pytest -rsf fbgemm_gpu/test/sparse/cumsum_test.py
pytest -rsf fbgemm_gpu/test/sparse/misc_ops_test.py
pytest -rsf fbgemm_gpu/test/sparse/permute_sparse_features_test.py
pytest -rsf fbgemm_gpu/test/sparse/reorder_batched_test.py
pytest -rsf fbgemm_gpu/test/tbe/utils/generate_vbe_metadata_test.py
pytest -rsf fbgemm_gpu/test/tbe/utils/split_embeddings_utils_test.py
```

## How to benchmark Intel® plugin for FBGEMM

The jagged operators are benchmarked with the `jagged-sweep` benchmark shipped in `fbgemm_xpu.bench`. It needs only `fbgemm-xpu` installed with the test dependencies, see [How to test Intel® plugin for FBGEMM](#how-to-test-intel-plugin-for-fbgemm). Run the sweep on a single device with:

```
ZE_AFFINITY_MASK=0 python -m fbgemm_xpu.bench.jagged_sweep --output jagged_tensor.csv
```

The sweep can be narrowed with `--batch-sizes`, `--max-lens`, `--embedding-dim`, `--dtypes` (`float32`, `bfloat16`, `float16`), `--families` (`jagged_to_padded_dense`, `jagged_2d_to_dense`, `dense_to_jagged`, `jagged_dense_elementwise_add_jagged_output`) and `--iters`. For a quick check that every case runs, use the `--smoke` flag. It selects a tiny grid and few iterations, so its timings are not meaningful:

```
python -m fbgemm_xpu.bench.jagged_sweep --smoke --output jagged_sweep_smoke.csv
```

See the [baselines README](packages/fbgemm-xpu/bench/baselines/README.md) for the sweep definition, how bandwidth is computed and the committed results.

The upstream FBGEMM jagged tensor benchmark script is also patched to time on XPU. Apply both FBGEMM patches to the pinned FBGEMM release (if you already prepared the FBGEMM checkout for testing, apply only the second patch):

```
export FBGEMM_XPU_PATH=$TORCHLIB_XPU_PATH/packages/fbgemm-xpu/
git clone --branch v1.9.0 https://github.com/pytorch/FBGEMM.git && cd FBGEMM
git am $FBGEMM_XPU_PATH/patches/0001-Add-XPU-support-to-fbgemm-tests.patch
git am $FBGEMM_XPU_PATH/patches/0002-Add-XPU-support-to-fbgemm-jagged-benchmark.patch
```

The benchmark patch can be reviewed [here](packages/fbgemm-xpu/patches/0002-Add-XPU-support-to-fbgemm-jagged-benchmark.patch). It changes only `fbgemm_gpu/bench/`, and CI enforces this. The patch only edits existing upstream code. New functions and classes go in the [`fbgemm_xpu.bench`](packages/fbgemm-xpu/src/fbgemm_xpu/bench/) package, which the patched script imports: the timing helper (`bench_utils`) and the XPU skips (`skips`).

The patch enables the upstream `device` benchmark on XPU. It only runs the sub-benchmarks whose operators have XPU implementations, and logs a reason for each one it skips:

```
ZE_AFFINITY_MASK=0 python fbgemm_gpu/bench/jagged_tensor_benchmark.py device --batch-size 8 --max-len 16
```

The other upstream benchmark cases don't use any XPU-implemented operator, so they are skipped on XPU. Run `python fbgemm_gpu/bench/jagged_tensor_benchmark.py --help` to list all cases and their options.

## Tips

### oneAPI compatibility

Use the following compatibility table when self-building the project and its dependencies:

| PyTorch | Torchvision | oneAPI         |
| ------- | ----------- | -------------- |
| 2.14    | 0.29        | [2026.1][2-14] |
| 2.13    | 0.28        | [2026.0][2-13] |
| 2.12    | 0.27        | [2025.3][2-12] |
| 2.11    | 0.26        | [2025.3][2-11] |
| 2.10    | 0.25        | [2025.3][2-10] |
| 2.9     | 0.24        | [2025.2][2-9]  |
| 2.8     | 0.23        | [2025.1][2-8]  |

[TorchCodec]: https://github.com/meta-pytorch/torchcodec
[FBGEMM]: https://github.com/pytorch/FBGEMM

[2-14]: https://www.intel.com/content/www/us/en/developer/articles/tool/pytorch-prerequisites-for-intel-gpu/2-14.html
[2-13]: https://www.intel.com/content/www/us/en/developer/articles/tool/pytorch-prerequisites-for-intel-gpu/2-13.html
[2-12]: https://www.intel.com/content/www/us/en/developer/articles/tool/pytorch-prerequisites-for-intel-gpu/2-12.html
[2-11]: https://www.intel.com/content/www/us/en/developer/articles/tool/pytorch-prerequisites-for-intel-gpu/2-11.html
[2-10]: https://www.intel.com/content/www/us/en/developer/articles/tool/pytorch-prerequisites-for-intel-gpu/2-10.html
[2-9]: https://www.intel.com/content/www/us/en/developer/articles/tool/pytorch-prerequisites-for-intel-gpu/2-9.html
[2-8]: https://www.intel.com/content/www/us/en/developer/articles/tool/pytorch-prerequisites-for-intel-gpu/2-8.html
