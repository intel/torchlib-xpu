# Intel XPU Plugin for FBGEMM

## Overview

[FBGEMM] is an optimized library for GEMMs and low-precision training. The Intel® XPU plugin for [FBGEMM] enables hardware acceleration for specific [FBGEMM] operators on Intel GPUs using SYCL kernels. Currently, acceleration is primarily targeted for DLRM v3 workloads.

To use Intel® XPU plugin for [FBGEMM], load it in your Python script and ensure tensors are on XPU device:

```python
import torch
import fbgemm_xpu

# Usage examples will be added as operators are integrated into this project
```

## Supported operators

This plugin provides Intel® XPU (SYCL) implementations for the following
operators, registered under the `torch.ops.fbgemm` namespace. Signatures
and behavior match [FBGEMM v1.8.0][fbgemm-tag] (pinned in
[pyproject.toml](pyproject.toml)); see the upstream
[FBGEMM sparse operators][fbgemm-sparse-ops] page for usage.

- [`asynchronous_complete_cumsum`][op-asynchronous_complete_cumsum]
- `asynchronous_exclusive_cumsum`
- `asynchronous_inclusive_cumsum`
- `invert_permute`
- [`permute_1D_sparse_data`][op-permute_1D_sparse_data]
- [`permute_2D_sparse_data`][op-permute_2D_sparse_data]
- `permute_2D_sparse_preallocated_out`
- [`block_bucketize_sparse_features`][op-block_bucketize_sparse_features]
- `block_bucketize_sparse_features_inference`
- `populate_bucketized_permute`

For operators without an upstream API entry, refer to their PyTorch schema
in [ops_registry.cpp](src/fbgemm_xpu/ops_registry.cpp).

## Supported hardware

Currently, this package has been tested only on Intel® Data Center GPU Max Series (Ponte Vecchio, PVC) GPUs.

## Installation

Pre-built wheels will be available on [PyPI](https://pypi.org) in the future.

For now, build from source:

* Install [uv]

* Install Intel oneAPI (DPC++ compiler `icpx`), version 2026.0

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

Known limitations will be documented as new FBGEMM operators are integrated into this project.

[FBGEMM]: https://github.com/pytorch/FBGEMM
[uv]: https://github.com/astral-sh/uv
[PVC]: https://www.intel.com/content/www/us/en/ark/products/series/232874/intel-data-center-gpu-max-series.html

[fbgemm-sparse-ops]: https://docs.pytorch.org/FBGEMM/fbgemm_gpu/python-api/sparse_ops.html
[fbgemm-tag]: https://github.com/pytorch/FBGEMM/tree/v1.8.0
[op-asynchronous_complete_cumsum]: https://docs.pytorch.org/FBGEMM/fbgemm_gpu/python-api/sparse_ops.html#torch.ops.fbgemm.asynchronous_complete_cumsum
[op-permute_1D_sparse_data]: https://docs.pytorch.org/FBGEMM/fbgemm_gpu/python-api/sparse_ops.html#torch.ops.fbgemm.permute_1D_sparse_data
[op-permute_2D_sparse_data]: https://docs.pytorch.org/FBGEMM/fbgemm_gpu/python-api/sparse_ops.html#torch.ops.fbgemm.permute_2D_sparse_data
[op-block_bucketize_sparse_features]: https://docs.pytorch.org/FBGEMM/fbgemm_gpu/python-api/sparse_ops.html#torch.ops.fbgemm.block_bucketize_sparse_features
