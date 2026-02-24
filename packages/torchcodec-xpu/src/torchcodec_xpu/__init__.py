# Copyright (c) Meta Platforms, Inc. and affiliates.
# Copyright (c) 2025 Dmitry Rogozhkin.

import ctypes
import importlib
import sys
from pathlib import Path

import torch
import torchcodec


def _get_extension_path(lib_name: str) -> str:
    extension_suffixes = []
    if sys.platform == "linux":
        extension_suffixes = importlib.machinery.EXTENSION_SUFFIXES
    #elif sys.platform in ("win32", "cygwin"):
    #    extension_suffixes = importlib.machinery.EXTENSION_SUFFIXES + [".dll", ".pyd"]
    else:
        raise NotImplementedError(f"{sys.platform = } is not not supported")
    loader_details = (
        importlib.machinery.ExtensionFileLoader,
        extension_suffixes,
    )

    extfinder = importlib.machinery.FileFinder(
        str(Path(__file__).parent), loader_details
    )
    ext_specs = extfinder.find_spec(lib_name)
    if ext_specs is None:
        raise ImportError(f"No spec found for {lib_name}")

    if ext_specs.origin is None:
        raise ImportError(f"Existing spec found for {lib_name} does not have an origin")

    return ext_specs.origin

def load_torchcodec_xpu_shared_library():
    exceptions = []
    ffmpeg_major_version = torchcodec.ffmpeg_major_version
    xpu_library_name = f"libtorchcodec_xpu{ffmpeg_major_version}"
    try:
        ctypes.CDLL(torchcodec.core_library_path)
        torch.ops.load_library(_get_extension_path(xpu_library_name))
        return
    except Exception as e:
        exceptions.append((ffmpeg_major_version, e))

    traceback = (
        "\n[start of libtorchcodec_xpu loading traceback]\n"
        + "\n".join(f"FFmpeg version {v}: {str(e)}" for v, e in exceptions)
        + "\n[end of libtorchcodec_xpu loading traceback]."
    )
    raise RuntimeError(
        f"""Could not load libtorchcodec_xpu. Likely causes:
          1. Missing dependencies. Such as FFmpeg, L0 or LibVA libraries.
          1. Intel extension for TorchCodec (libtorchcodec_xpu) is not compatible
             with this version of TorchCodec.
          2. The PyTorch version ({torch.__version__}) is not compatible with
             this version of TorchCodec.
          3. Another runtime dependency; see exceptions below.
        The following exceptions were raised as we tried to load libtorchcodec_xpu:
        """
        f"{traceback}"
    )

load_torchcodec_xpu_shared_library()

