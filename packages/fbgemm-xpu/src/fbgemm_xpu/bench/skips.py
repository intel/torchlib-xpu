# Copyright (c) 2026 Intel Corporation. All Rights Reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""Skips for the cases of the patched upstream jagged tensor benchmark.

The patched ``jagged_tensor_benchmark.py`` times on XPU only, so the cases and
sub-benchmarks that drive no XPU-implemented operator are replaced with a
logged skip instead of being removed. Run them from an unpatched checkout.
"""

import functools
import logging
from collections.abc import Callable

# Sub-benchmarks of the `device` case that do not run on XPU, with the reason.
XPU_SKIPPED_SUB_BENCHMARKS: dict[str, str] = {
    "bench_jagged_dense_dense_elementwise_add_jagged_output": "jagged_dense_dense_elementwise_add_jagged_output_forward has no XPU implementation",
    "bench_jagged_1d_to_dense": "jagged_1d_to_dense is not an XPU-registered operator; its kernel, jagged_to_padded_dense, is timed by jagged-sweep",
    "bench_dense_to_jagged_1d": "1D (D = 1) form of dense_to_jagged, which jagged-sweep times; upstream also ignores --elem-type here",
}


def skipped_on_xpu(reason: str) -> Callable[[Callable[..., None]], Callable[..., None]]:
    """Replace a case that drives no XPU-implemented operator with a logged skip."""

    def wrap(fn: Callable[..., None]) -> Callable[..., None]:
        @functools.wraps(fn)
        def skipped(*args: object, **kwargs: object) -> None:
            logging.info(f"SKIPPED {fn.__name__} on xpu: {reason}")

        return skipped

    return wrap


def drop_xpu_skipped_sub_benchmarks(
    benchmarks: dict[str, Callable[..., None]], only: str | None
) -> str | None:
    """Remove the sub-benchmarks that do not run on XPU from ``benchmarks``.

    ``benchmarks`` is modified in place, and a skip is logged for each removed
    sub-benchmark that was requested. Returns ``only`` without the removed
    names, so upstream's validation still rejects unknown names. The result is
    an empty string if every requested sub-benchmark was removed, and None if
    ``only`` was empty, which means run all.
    """
    requested = [n.strip() for n in only.split(",")] if only else list(benchmarks)
    for name, reason in XPU_SKIPPED_SUB_BENCHMARKS.items():
        del benchmarks[name]
        if name in requested:
            logging.info(f"SKIPPED {name} on xpu: {reason}")
    if not only:
        return None
    return ",".join(n for n in requested if n not in XPU_SKIPPED_SUB_BENCHMARKS)
