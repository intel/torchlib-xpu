# fbgemm-xpu

Intel XPU plugin package for FBGEMM operators.

## Build from source

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

[uv]: https://github.com/astral-sh/uv
