# Copyright (c) 2026 Intel Corporation. All Rights Reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""Deterministic checks for dma-buf fd handling and multi-threaded decoding.

The plugin imports the dma-buf exported by VAAPI for every decoded frame. These
tests check the observable effects of doing that correctly: no leaked file
descriptors, no descriptors closed behind other threads' backs, and the same
frames when decoding concurrently as when decoding alone.
"""

import gc
import os
import threading
from concurrent.futures import ThreadPoolExecutor

import pytest
import torch
from torchcodec.decoders import VideoDecoder

NUM_THREADS = 8

# Rounds of concurrent decoding in the descriptor test. The failure it detects
# needs a narrow interleaving, so more rounds raise the chance of hitting it.
NUM_ROUNDS = 5


def _open_fd_count() -> int:
    return len(os.listdir("/proc/self/fd"))


def _decode_range(path: str, start: int, stop: int) -> torch.Tensor:
    decoder = VideoDecoder(path, device="xpu", seek_mode="approximate")
    return decoder.get_frames_in_range(start, stop).data.cpu()


def test_decoding_does_not_leak_fds(xpu_clip: str) -> None:
    # Each frame's dma-buf fd is closed either by the driver or, if the driver
    # does not, by the plugin. Decoding the same frames repeatedly must not
    # grow the number of open descriptors.
    def decode_round() -> int:
        _decode_range(xpu_clip, 0, 30)
        gc.collect()
        return _open_fd_count()

    # The first round may open descriptors that stay open for the process
    # lifetime (render node, Level Zero), so only compare later rounds.
    baseline = decode_round()
    counts = [decode_round() for _ in range(3)]

    # A leak of one fd per frame would add 30 per round.
    assert counts == [baseline] * len(counts)  # nosec B101


def test_decoding_does_not_close_fds_of_other_threads(xpu_clip: str) -> None:
    # If the plugin closed a descriptor number it no longer owns, that number
    # may by then belong to another thread. Bystander threads emulate such a
    # thread: they keep opening a descriptor, using it and closing it, and
    # report an error if it has been closed by someone else in the meantime.
    stop = threading.Event()
    errors: list[str] = []

    def bystander() -> None:
        while not stop.is_set():
            fd = os.open(os.devnull, os.O_RDONLY)
            try:
                os.fstat(fd)
                os.close(fd)
            except OSError as e:
                errors.append(f"fd {fd}: {e}")

    bystanders = [threading.Thread(target=bystander) for _ in range(4)]
    for thread in bystanders:
        thread.start()
    try:
        for _ in range(NUM_ROUNDS):
            with ThreadPoolExecutor(NUM_THREADS) as pool:
                futures = [
                    pool.submit(_decode_range, xpu_clip, 0, 30)
                    for _ in range(NUM_THREADS)
                ]
                for future in futures:
                    future.result()
    finally:
        stop.set()
        for thread in bystanders:
            thread.join()

    assert not errors, (  # nosec B101
        f"descriptors were closed under other threads: {errors[:5]}"
    )


@pytest.mark.parametrize("start,stop", [(0, 30), (45, 75), (90, 120)])
def test_concurrent_decode_matches_single_thread(
    xpu_clip: str, start: int, stop: int
) -> None:
    reference = _decode_range(xpu_clip, start, stop)

    with ThreadPoolExecutor(NUM_THREADS) as pool:
        futures = [
            pool.submit(_decode_range, xpu_clip, start, stop)
            for _ in range(NUM_THREADS)
        ]
        results = [future.result() for future in futures]

    for result in results:
        torch.testing.assert_close(result, reference, rtol=0, atol=0)
