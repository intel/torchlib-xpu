# Copyright (c) Meta Platforms, Inc. and affiliates. All rights reserved.
# Copyright (c) 2026 Intel Corporation. All Rights Reserved.
# SPDX-License-Identifier: BSD-3-Clause

# Import torch first so libtorch shared libraries are mapped into the process
# before _C loads, then fbgemm_gpu so that all "fbgemm" operator schemas are
# registered before _C provides the XPU implementations via TORCH_LIBRARY_IMPL.
import torch  # noqa: F401, E402, I001
import fbgemm_gpu  # noqa: F401, E402

# Import the compiled C extension (_C) which contains the registered operators.
# Loading failures must remain visible because _C owns the XPU operator schemas.
from . import _C as _C

# Load the training / codegen extension that registers XPU and AutogradXPU
# implementations for the embedding lookup operators
# (dense_embedding_codegen_lookup_function, split_embedding_codegen_lookup_*).
# It must be imported after _C so the operator schemas declared there are
# already visible when the TORCH_LIBRARY_IMPL static initialisers run.
from . import _C_training as _C_training

__all__ = ["_C", "_C_training", "__version__"]

try:
    from ._version import __version__
except ModuleNotFoundError:
    try:
        from importlib.metadata import PackageNotFoundError, version
        __version__ = version("fbgemm-xpu")
    except (ImportError, PackageNotFoundError):
        __version__ = "unknown"
