#!/usr/bin/env bash
# Copyright (c) 2026 Intel Corporation. All Rights Reserved.
# SPDX-License-Identifier: BSD-3-Clause
#
# Installs TorchRec from a checkout and the pinned dependencies of DLRM-v3
# into the active environment, which already has fbgemm-xpu.
#
# Usage: install-dependencies.sh <torchrec checkout>
# Environment: NUMPY_VERSION, PANDAS_VERSION

set -euo pipefail

: "${NUMPY_VERSION:?}" "${PANDAS_VERSION:?}"
torchrec=$(cd "${1:?usage: install-dependencies.sh <torchrec checkout>}" && pwd)
here=$(cd "$(dirname "$0")" && pwd)

torch_version=$(python -c "import torch; print(torch.__version__)")
pip install --no-cache-dir \
  "torch==$torch_version" \
  "numpy==$NUMPY_VERSION" \
  "pandas==$PANDAS_VERSION" \
  -r "$here/requirements.txt" \
  --extra-index-url https://download.pytorch.org/whl/xpu

# --no-deps: TorchRec requires fbgemm-gpu, the CUDA build; the CPU build
# fbgemm-gpu-cpu is installed with fbgemm-xpu instead.
pip install --no-cache-dir --no-deps --no-build-isolation "$torchrec"

if pip show fbgemm-gpu >/dev/null 2>&1; then
  echo "The CUDA build of FBGEMM (fbgemm-gpu) must not be installed" >&2
  exit 1
fi
python -c "import torch; assert torch.__version__ == '$torch_version', torch.__version__"
