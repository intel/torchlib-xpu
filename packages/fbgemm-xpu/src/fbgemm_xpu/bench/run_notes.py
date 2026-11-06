# Copyright (c) 2026 Intel Corporation. All Rights Reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""Run notes shared by the fbgemm_xpu benchmarks.

A benchmark writes its timings to a plain CSV, which GitHub renders as a
table, and the run's command, date, device, driver, versions and definition to
a Markdown file of the same name next to it. Two runs are comparable only if
these notes match.
"""

import datetime
import os
import sys
from pathlib import Path

import fbgemm_gpu
import torch

import fbgemm_xpu
from fbgemm_xpu.bench.bench_utils import gpu_lead_method


def run_metadata(module: str, **spec: object) -> dict[str, object]:
    """The environment of this run, followed by the benchmark's ``spec``."""
    return {
        "command": " ".join(["python", "-m", module, *sys.argv[1:]]),
        "date_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(
            timespec="seconds"
        ),
        "device": torch.xpu.get_device_name(),
        "driver": torch.xpu.get_device_properties().driver_version,
        "ze_affinity_mask": os.environ.get("ZE_AFFINITY_MASK", ""),
        "torch": torch.__version__,
        "fbgemm_gpu": getattr(fbgemm_gpu, "__version__", "unknown"),
        "fbgemm_xpu": fbgemm_xpu.__version__,
        "timing": "median of XPU event times, 2 warm-ups, "
        f"{2 * torch.xpu.get_device_properties().last_level_cache_size >> 20} MB "
        "cache flush (2x last-level cache, written, then a second buffer "
        "read so the cache is left clean) per iteration, then "
        f"{gpu_lead_method()} sized from the fastest warm-up "
        "so the GPU does not wait for the host in the timed window",
        **spec,
    }


def notes_path(output: str) -> Path:
    return Path(output).with_suffix(".md")


def notes_markdown(
    csv_name: str,
    subject: str,
    metadata: dict[str, object],
    fields: list[str],
) -> str:
    """The run notes for ``csv_name``, timings of ``subject``.

    One table row per metadata key; the CSV itself carries no comments so that
    it renders as a table on GitHub.
    """
    lines = [
        f"# {csv_name}",
        "",
        f"Timings of {subject} written by `{metadata['command']}`.",
        f"Columns: {', '.join(f'`{f}`' for f in fields)}.",
        "",
        "| Key | Value |",
        "| --- | --- |",
    ]
    for key, value in metadata.items():
        # Pipes and newlines in values would break the table.
        cell = str(value).replace("|", "\\|").replace("\n", " ")
        lines.append(f"| {key} | {cell} |")
    return "\n".join(lines) + "\n"
