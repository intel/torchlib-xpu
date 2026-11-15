# Copyright (c) 2026 Intel Corporation. All Rights Reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""Multi-threaded dataloader-style workload test for the residency fix.

The test runs the workload of torchcodec's benchmarks/decoders/
generate_readme_data.py "concurrency" case in a subprocess: a thread pool of
workers, each task creating its own decoder, decoding a batch of frames played
at uniform timestamps, and resizing them on the XPU. Failures here are crashes,
aborts, hangs and GPU resets, which cannot be caught in-process, hence the
subprocess and the timeout.

The failure is timing dependent. The test is therefore a probabilistic
detector: it reliably passes with the fix, and it fails in most runs without
it. A failure means the fix it names is missing or broken.
"""

import os
import subprocess  # nosec B404 - runs this module's workload with fixed arguments
import sys

from conftest import CLIP_1080P_DURATION

NUM_THREADS = 10
BATCH_SIZE = 50
NUM_FRAMES = 10
RESIZE = 256
TIMEOUT_SECONDS = 240
REPETITIONS_RESIDENCY = 5

WORKLOAD = """
import sys
from concurrent.futures import ThreadPoolExecutor

import torch
from torchcodec.decoders import VideoDecoder
from torchvision.transforms import v2

path = sys.argv[1]
duration = float(sys.argv[2])
num_threads, batch_size, num_frames, resize, repetitions = map(int, sys.argv[3:8])
pts_list = [i * duration / num_frames for i in range(num_frames)]


def decode_and_resize():
    decoder = VideoDecoder(
        path, num_ffmpeg_threads=1, device="xpu", seek_mode="approximate"
    )
    frames = decoder.get_frames_played_at(pts_list)
    return v2.functional.resize(frames.data, (resize, resize))


for _ in range(repetitions):
    with ThreadPoolExecutor(num_threads) as pool:
        futures = [pool.submit(decode_and_resize) for _ in range(batch_size)]
        for future in futures:
            assert len(future.result()) > 0
torch.xpu.synchronize()
"""


def _run_workload(path: str, repetitions: int, use_sycl_kernels: str) -> None:
    env = {**os.environ, "USE_SYCL_KERNELS": use_sycl_kernels}
    args = [
        path,
        str(CLIP_1080P_DURATION),
        str(NUM_THREADS),
        str(BATCH_SIZE),
        str(NUM_FRAMES),
        str(RESIZE),
        str(repetitions),
    ]
    try:
        result = subprocess.run(  # nosec B603
            [sys.executable, "-c", WORKLOAD, *args],
            env=env,
            capture_output=True,
            text=True,
            timeout=TIMEOUT_SECONDS,
        )
    except subprocess.TimeoutExpired as e:
        raise AssertionError(f"decoding hung for {TIMEOUT_SECONDS}s: {e.stderr}") from e

    assert result.returncode == 0, (  # nosec B101
        f"workload failed with exit code {result.returncode}:\n{result.stderr[-2000:]}"
    )


def test_dataloader_workload_without_resident_import(xpu_clip_1080p: str) -> None:
    """Detects a missing zeContextMakeMemoryResident() after the dma-buf import.

    Without it the driver sometimes never binds an imported allocation, the next
    kernel using it page-faults and the queue is reset. Observed as
    UR_RESULT_ERROR_DEVICE_LOST or UR_RESULT_ERROR_OUT_OF_RESOURCES, in a
    fraction of runs, so the workload is repeated.
    """
    _run_workload(xpu_clip_1080p, repetitions=REPETITIONS_RESIDENCY, use_sycl_kernels="1")
