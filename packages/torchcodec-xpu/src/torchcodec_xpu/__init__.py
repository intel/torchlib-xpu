# Copyright (c) Meta Platforms, Inc. and affiliates.
# Copyright (c) 2025 Dmitry Rogozhkin.

import ctypes
import importlib
import traceback

import torch
import torchcodec


def _get_extension_path(lib_name: str) -> str:
    spec = importlib.util.find_spec(lib_name)
    if spec is None or spec.origin is None:
        raise ImportError(f"No spec found for {lib_name}")
    return spec.origin

def load_torchcodec_xpu_shared_library():
    exceptions = []
    ffmpeg_major_version = torchcodec.ffmpeg_major_version
    xpu_library_name = f"torchcodec_xpu.xpu_ops{ffmpeg_major_version}"
    try:
        ctypes.CDLL(torchcodec.core_library_path)
        torch.ops.load_library(_get_extension_path(xpu_library_name))
        return
    except Exception:
        # Capture the full traceback for this exception
        exc_traceback = traceback.format_exc()
        exceptions.append((ffmpeg_major_version, exc_traceback))

    traceback_info = (
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
        f"{traceback_info}"
    )

load_torchcodec_xpu_shared_library()

