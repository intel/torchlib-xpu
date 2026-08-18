# Copyright (c) Meta Platforms, Inc. and affiliates. All rights reserved.
# Copyright (c) 2026 Intel Corporation. All Rights Reserved.
# SPDX-License-Identifier: BSD-3-Clause

# Import fbgemm_gpu first so that all "fbgemm" operator schemas are registered
# before _C loads the XPU implementations via TORCH_LIBRARY_IMPL.
import fbgemm_gpu  # noqa: F401, E402

# Promote fbgemm_gpu's native libraries to the global symbol namespace.
#
# Some XPU implementations reuse device-agnostic host helpers that are exported
# (DLL_PUBLIC) by the fbgemm-gpu-cpu wheel -- e.g. get_infos_metadata reuses
# get_info_B_num_bits_from_T from fbgemm_gpu_tbe_utils.so. fbgemm_gpu loads its
# .so files with RTLD_LOCAL, so those symbols are not visible to _C and it would
# fail to load with "undefined symbol". Re-opening them with RTLD_GLOBAL (a
# no-op reload that only updates the flags for already-mapped libraries) makes
# the exported helpers resolvable when _C is loaded below.
import ctypes as _ctypes  # noqa: E402
import glob as _glob  # noqa: E402
import os as _os  # noqa: E402

for _so in _glob.glob(_os.path.join(_os.path.dirname(fbgemm_gpu.__file__), "*.so")):
    try:
        _ctypes.CDLL(_so, mode=_ctypes.RTLD_GLOBAL)
    except OSError:
        pass

# Import the compiled C extension (_C) which contains the registered operators.
# If native dependencies (for example libtorch.so) are unavailable, keep import
# working so metadata like __version__ remains accessible.
try:
    from . import _C as _C
except ImportError:
    _C = None

__all__ = ["_C", "__version__"]

try:
    from ._version import __version__
except ModuleNotFoundError:
    try:
        from importlib.metadata import PackageNotFoundError, version
        __version__ = version("fbgemm-xpu")
    except (ImportError, PackageNotFoundError):
        __version__ = "unknown"
