# Copyright (c) Meta Platforms, Inc. and affiliates. All rights reserved.
# Copyright (c) 2026 Intel Corporation. All Rights Reserved.
# SPDX-License-Identifier: BSD-3-Clause

# Import the compiled C extension (_C) which contains the registered operators.
# If native dependencies (for example libtorch.so) are unavailable, keep import
# working so metadata like __version__ remains accessible.
try:
    from . import _C as _C
except ImportError:
    _C = None

from . import ops as ops

__all__ = ["_C", "ops", "__version__"]

try:
    from ._version import __version__
except ModuleNotFoundError:
    try:
        from importlib.metadata import PackageNotFoundError, version
        __version__ = version("fbgemm-xpu")
    except (ImportError, PackageNotFoundError):
        __version__ = "unknown"
