# Intel XPU Plugin for FBGEMM

## Overview

[FBGEMM] is an optimized library for GEMMs and low-precision training. The Intel® XPU plugin for [FBGEMM] enables hardware acceleration for specific [FBGEMM] operators on Intel GPUs using SYCL kernels. Currently, acceleration is primarily targeted for DLRM v3 workloads.

To use Intel® XPU plugin for [FBGEMM], load it in your Python script and ensure
tensors are on an XPU device:

```python
import torch
import fbgemm_xpu
```

## Supported operators

This plugin provides Intel® XPU (SYCL) implementations for the following
operators, registered under the `torch.ops.fbgemm` namespace.

* Implemented [FBGEMM sparse operators][fbgemm-sparse-ops]:

  - [`asynchronous_complete_cumsum`][op-asynchronous_complete_cumsum]
  - [`block_bucketize_sparse_features`][op-block_bucketize_sparse_features]
  - [`expand_into_jagged_permute`][op-expand_into_jagged_permute]
  - [`permute_1D_sparse_data`][op-permute_1D_sparse_data]
  - [`permute_2D_sparse_data`][op-permute_2D_sparse_data]

* Training lookup operators supported through direct calls (see the
  [FBGEMM TBE training documentation][fbgemm-tbe-training]):

  - `torch.ops.fbgemm.dense_embedding_codegen_lookup_function`: no-bag,
    unweighted lookup with dense autograd;
  - `torch.ops.fbgemm.split_embedding_codegen_lookup_rowwise_adagrad_function_pt2`:
    no-bag, unweighted lookup with rowwise-Adagrad in-place update.

  Both entry points are validated with:

  - one- and two-table layouts with one or two batches per table and a uniform
    embedding dimension;
  - FP32 and FP16 weight storage;
  - small and general forward kernels (`D=4` and `D=36`);
  - warp and CTA backward/update paths, including 32 repeated indices for one
    embedding row.

  These lookup paths are currently validated in eager mode only. FakeTensor
  tracing and `torch.compile` are not supported.

* Quantized inference lookup implementation:

  - `torch.ops.fbgemm.int_nbit_split_embedding_codegen_lookup_function`:
    eager, unweighted, no-bag INT4/INT8 with uniform dimensions and DEVICE
    storage. CPU-prepared rows use the FBGEMM packed INT4/INT8 layout.
  - Supports positive dimensions divisible by four, int32/int64 indices and
    offsets, FP32/FP16/BF16 output shaped `[indices.numel(), D]`, and
    power-of-two row alignment from 1 through 128 (default 16).
  - Does not support pruning/index remapping, mixed dimensions, `torch.compile`,
    or FakeTensor execution.

The following operators are also implemented but do not constitute
public documented FBGEMM API. These are extra variants, helpers, or utility
operators alongside the operators above. You can find their exact signature in
[ops_registry.cpp](src/fbgemm_xpu/ops_registry.cpp):

- `asynchronous_exclusive_cumsum`
- `asynchronous_inclusive_cumsum`
- `block_bucketize_sparse_features_inference`
- `bounds_check_indices`
- `dense_to_jagged`
- `dense_to_jagged_forward`
- `get_infos_metadata`
- `invert_permute`
- `jagged_2d_to_dense`
- `jagged_dense_elementwise_add_jagged_output`
- `jagged_index_select_2d_forward`
- `jagged_to_padded_dense`
- `jagged_to_padded_dense_backward`
- `jagged_to_padded_dense_forward`
- `permute_2D_sparse_preallocated_out`
- `populate_bucketized_permute`
- `reorder_batched_ad_indices`
- `reorder_batched_ad_lengths`

## Supported hardware

Currently, this package is tested on Intel® Data Center GPU Max Series
(Ponte Vecchio, PVC) GPUs and in CI on BMG hardware.

## Installation

Pre-built wheels will be available on [PyPI](https://pypi.org) in the future.

For now, build from source:

* Install [uv]

* Install Intel oneAPI (DPC++ compiler `icpx`), version 2026.1

* Clone the repository:

```bash
git clone https://github.com/intel/torchlib-xpu.git && cd torchlib-xpu
```

* Create and activate a virtual environment:

```bash
uv venv
source .venv/bin/activate
```

* Build and install `fbgemm-xpu`:

```bash
uv pip install -e packages/fbgemm-xpu \
  --index https://download.pytorch.org/whl/xpu
```

* (Optional) Install test dependencies:

```bash
uv pip install -e "packages/fbgemm-xpu[test]" \
  --index https://download.pytorch.org/whl/xpu
```

* Get installed package version:

```bash
python -c "import fbgemm_xpu; print(fbgemm_xpu.__version__)"
```

## Environment variables

Environment variables will be added as new FBGEMM operators are integrated into this project.

## Known limitations

The lookup operators do not currently support:

- pooled lookup (`PoolingMode.SUM` or `PoolingMode.MEAN`);
- weighted lookup;
- variable batch embeddings (VBE);
- global weight decay (GWD);
- cache-backed lookup.

The pristine FBGEMM 1.9.0 high-level training frontend does not expose
`ComputeDevice.XPU`. Constructing
`SplitTableBatchedEmbeddingBagsCodegen` for XPU is therefore unavailable in
this release. The upstream high-level TBE tests therefore do not exercise these
lookup operators on XPU. Instead, CI runs
[`test_lookup_ops.py`](tests/test_lookup_ops.py), which calls the supported
`torch.ops.fbgemm` methods directly on XPU. The separately patched upstream
FBGEMM tests cover the existing non-lookup operators.

`bounds_check_indices` implements version 1 only. Version 2 and
`prefetch_pipeline=True` are not implemented on XPU.

[FBGEMM]: https://github.com/pytorch/FBGEMM
[uv]: https://github.com/astral-sh/uv
[PVC]: https://www.intel.com/content/www/us/en/ark/products/series/232874/intel-data-center-gpu-max-series.html

[fbgemm-sparse-ops]: https://docs.pytorch.org/FBGEMM/fbgemm_gpu/python-api/sparse_ops.html
[fbgemm-tbe-training]: https://docs.pytorch.org/FBGEMM/fbgemm_gpu/python-api/tbe_ops_training.html
[op-asynchronous_complete_cumsum]: https://docs.pytorch.org/FBGEMM/fbgemm_gpu/python-api/sparse_ops.html#torch.ops.fbgemm.asynchronous_complete_cumsum
[op-permute_1D_sparse_data]: https://docs.pytorch.org/FBGEMM/fbgemm_gpu/python-api/sparse_ops.html#torch.ops.fbgemm.permute_1D_sparse_data
[op-permute_2D_sparse_data]: https://docs.pytorch.org/FBGEMM/fbgemm_gpu/python-api/sparse_ops.html#torch.ops.fbgemm.permute_2D_sparse_data
[op-block_bucketize_sparse_features]: https://docs.pytorch.org/FBGEMM/fbgemm_gpu/python-api/sparse_ops.html#torch.ops.fbgemm.block_bucketize_sparse_features
[op-expand_into_jagged_permute]: https://docs.pytorch.org/FBGEMM/fbgemm_gpu/python-api/sparse_ops.html#torch.ops.fbgemm.expand_into_jagged_permute
