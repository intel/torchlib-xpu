# Intel XPU Plugin for FBGEMM

## Overview

[FBGEMM] is an optimized library for GEMMs and low-precision training. The Intel® XPU plugin for [FBGEMM] enables hardware acceleration for specific [FBGEMM] operators on Intel GPUs using SYCL kernels. Currently, acceleration is primarily targeted for DLRM v3 workloads.

To use Intel® XPU plugin for [FBGEMM], load it in your Python script and ensure tensors are on XPU device:

```python
import torch
import fbgemm_xpu

# Usage examples will be added as operators are integrated into this project
```

## Supported hardware

Currently, this package has been tested only on Intel® Data Center GPU Max Series (Ponte Vecchio, PVC) GPUs.

## Installation

Pre-built wheels will be available on [PyPI](https://pypi.org) in the future.

For now, build from source:

* Install [uv]

* Install Intel oneAPI (DPC++ compiler `icpx`), version 2025.3 or newer

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

