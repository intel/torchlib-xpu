# Copyright (c) 2026 Intel Corporation. All Rights Reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""The jagged-sweep benchmark for the jagged operators registered on XPU.

It times every jagged operator registered on XPU, forward and backward,
against one shape sweep. Each shape is checked against CPU before it is timed,
so an incorrect result cannot be recorded as a fast one. Run it with::

    python -m fbgemm_xpu.bench.jagged_sweep --output jagged_tensor.csv

The CSV holds the timings only. The command, date, device, driver, versions
and sweep definition go to a Markdown file next to it, ``jagged_tensor.md``
here, so the CSV stays a plain table that GitHub renders.
"""

import csv
import datetime
import functools
import logging
import os
import sys
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import click
import fbgemm_gpu
import torch

import fbgemm_xpu
from fbgemm_xpu.bench.bench_utils import benchmark_torch_function, gpu_lead_method

_MODULE = "fbgemm_xpu.bench.jagged_sweep"


def _nbytes(t: torch.Tensor) -> int:
    return t.numel() * t.element_size()


@dataclass
class SweepFamily:
    """An operator family timed by jagged-sweep.

    ``fwd`` maps the inputs (``values``, ``offsets``, ``dense``), ``max_len``
    and ``total_L`` to the output; ``grad_inputs`` names the inputs the backward
    produces gradients for. Bytes are the logical bytes read plus written by
    one call: jagged tensors count values and offsets, dense tensors count the
    full ``B * max_len * D``. The forward figures follow the formulas of the
    corresponding upstream `device` sub-benchmarks.
    """

    fwd: Callable[[dict[str, torch.Tensor], int, int], torch.Tensor]
    grad_inputs: tuple[str, ...]
    fwd_bytes: Callable[[dict[str, torch.Tensor], torch.Tensor], int]
    bwd_bytes: Callable[[dict[str, torch.Tensor], torch.Tensor], int]


def _jagged_to_dense_bytes(i: dict[str, torch.Tensor], out: torch.Tensor) -> int:
    # Forward reads the jagged values and writes the dense output; backward
    # reads the dense gradient and writes the jagged one, the same amount.
    return _nbytes(i["values"]) + _nbytes(i["offsets"]) + _nbytes(out)


SWEEP_FAMILIES: dict[str, SweepFamily] = {
    "jagged_to_padded_dense": SweepFamily(
        fwd=lambda i, max_len, total_L: torch.ops.fbgemm.jagged_to_padded_dense(
            i["values"], [i["offsets"]], [max_len], 0.0
        ),
        grad_inputs=("values",),
        fwd_bytes=_jagged_to_dense_bytes,
        bwd_bytes=_jagged_to_dense_bytes,
    ),
    "jagged_2d_to_dense": SweepFamily(
        fwd=lambda i, max_len, total_L: torch.ops.fbgemm.jagged_2d_to_dense(
            i["values"], i["offsets"], max_len
        ),
        grad_inputs=("values",),
        fwd_bytes=_jagged_to_dense_bytes,
        bwd_bytes=_jagged_to_dense_bytes,
    ),
    "dense_to_jagged": SweepFamily(
        fwd=lambda i, max_len, total_L: torch.ops.fbgemm.dense_to_jagged(
            i["dense"], [i["offsets"]], total_L
        )[0],
        grad_inputs=("dense",),
        fwd_bytes=lambda i, out: _nbytes(i["dense"])
        + _nbytes(out)
        + _nbytes(i["offsets"]),
        bwd_bytes=lambda i, out: _nbytes(out)
        + _nbytes(i["offsets"])
        + _nbytes(i["dense"]),
    ),
    "jagged_dense_elementwise_add_jagged_output": SweepFamily(
        fwd=lambda i, max_len, total_L: torch.ops.fbgemm.jagged_dense_elementwise_add_jagged_output(
            i["values"], [i["offsets"]], i["dense"]
        )[0],
        grad_inputs=("values", "dense"),
        # Reads x and the matching part of y, writes the output.
        fwd_bytes=lambda i, out: _nbytes(i["offsets"]) + 3 * _nbytes(i["values"]),
        # The values gradient is the output gradient itself; only the dense
        # gradient is computed, by padding the output gradient to dense.
        bwd_bytes=lambda i, out: _nbytes(out)
        + _nbytes(i["offsets"])
        + _nbytes(i["dense"]),
    ),
}

SWEEP_DTYPES: dict[str, torch.dtype] = {
    "float32": torch.float32,
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
}


def _int_list(value: str) -> list[int]:
    return [int(v) for v in value.split(",")]


def _sweep_metadata(**spec: object) -> dict[str, object]:
    return {
        "command": " ".join(["python", "-m", _MODULE, *sys.argv[1:]]),
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
        "lengths": "uniform integers in [0, max_len], one draw per shape",
        **spec,
    }


def _metadata_path(output: str) -> Path:
    return Path(output).with_suffix(".md")


def _metadata_markdown(csv_name: str, metadata: dict[str, object], fields: list[str]) -> str:
    # One table row per key; the CSV itself carries no comments so that it
    # renders as a table on GitHub. Pipes in values would break the table.
    lines = [
        f"# {csv_name}",
        "",
        f"Timings of the XPU jagged operators written by `{metadata['command']}`.",
        f"Columns: {', '.join(f'`{f}`' for f in fields)}.",
        "",
        "| Key | Value |",
        "| --- | --- |",
    ]
    for key, value in metadata.items():
        cell = str(value).replace("|", "\\|")
        lines.append(f"| {key} | {cell} |")
    return "\n".join(lines) + "\n"


def _check_against_cpu(
    name: str,
    family: SweepFamily,
    cpu: dict[str, torch.Tensor],
    xpu: dict[str, torch.Tensor],
    max_len: int,
    total_L: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run forward and backward on both devices and require them to match.

    Returns the XPU output, with its graph retained, and the output gradient,
    for timing the backward.
    """
    out_cpu = family.fwd(cpu, max_len, total_L)
    out_xpu = family.fwd(xpu, max_len, total_L)
    torch.testing.assert_close(out_xpu.cpu(), out_cpu, msg=lambda m: f"{name} fwd: {m}")

    grad_out = torch.rand_like(out_cpu)
    grads_cpu = torch.autograd.grad(
        out_cpu, [cpu[n] for n in family.grad_inputs], grad_out
    )
    grad_out_xpu = grad_out.xpu()
    grads_xpu = torch.autograd.grad(
        out_xpu,
        [xpu[n] for n in family.grad_inputs],
        grad_out_xpu,
        retain_graph=True,
    )
    for n, g_cpu, g_xpu in zip(family.grad_inputs, grads_cpu, grads_xpu):
        torch.testing.assert_close(
            g_xpu.cpu(), g_cpu, msg=lambda m, n=n: f"{name} bwd d{n}: {m}"
        )
    return out_xpu, grad_out_xpu


@click.command(name="jagged-sweep")
@click.option("--batch-sizes", type=str, default="32,128,512,2048")
@click.option("--max-lens", type=str, default="16,64,200,256,512,1024")
@click.option(
    "--embedding-dim",
    type=int,
    default=128,
    help="Inner dimension D. Default is the DLRM v3 HSTU embedding width.",
)
@click.option("--dtypes", type=str, default=",".join(SWEEP_DTYPES))
@click.option("--families", type=str, default=",".join(SWEEP_FAMILIES))
@click.option("--iters", type=int, default=100)
@click.option("--seed", type=int, default=0)
@click.option(
    "--smoke",
    is_flag=True,
    default=False,
    help="Tiny grid and few iterations: exercises every case, timings are not meaningful.",
)
@click.option("--output", type=click.Path(dir_okay=False), default="jagged_sweep.csv")
def jagged_sweep(
    batch_sizes: str,
    max_lens: str,
    embedding_dim: int,
    dtypes: str,
    families: str,
    iters: int,
    seed: int,
    smoke: bool,
    output: str,
) -> None:
    if not torch.xpu.is_available():
        raise click.UsageError("jagged-sweep requires an XPU device.")
    if smoke:
        batch_sizes, max_lens, iters = "4,16", "8,33", 5
    sweep_batch_sizes = _int_list(batch_sizes)
    sweep_max_lens = _int_list(max_lens)
    sweep_dtypes = [SWEEP_DTYPES[d] for d in dtypes.split(",")]
    sweep_families = {f: SWEEP_FAMILIES[f] for f in families.split(",")}

    torch.manual_seed(seed)
    metadata = _sweep_metadata(
        batch_sizes=batch_sizes,
        max_lens=max_lens,
        embedding_dim=embedding_dim,
        dtypes=dtypes,
        families=families,
        iters=iters,
        seed=seed,
    )
    fields = [
        "family",
        "direction",
        "dtype",
        "batch_size",
        "max_len",
        "embedding_dim",
        "total_lengths",
        "time_us",
        "bytes",
        "gb_per_s",
    ]
    rows = []
    for dtype in sweep_dtypes:
        for batch_size in sweep_batch_sizes:
            for max_len in sweep_max_lens:
                lengths = torch.randint(max_len + 1, (batch_size,))
                total_L = int(lengths.sum())
                cpu_inputs = {
                    "values": torch.rand(total_L, embedding_dim).to(dtype),
                    "offsets": torch.ops.fbgemm.asynchronous_complete_cumsum(lengths),
                    "dense": torch.rand(batch_size, max_len, embedding_dim).to(dtype),
                }
                for name, family in sweep_families.items():
                    cpu = {
                        k: v.clone().requires_grad_(k in family.grad_inputs)
                        for k, v in cpu_inputs.items()
                    }
                    xpu = {
                        k: v.xpu().requires_grad_(k in family.grad_inputs)
                        for k, v in cpu_inputs.items()
                    }
                    out, grad_out = _check_against_cpu(
                        name, family, cpu, xpu, max_len, total_L
                    )
                    grad_targets = [xpu[n] for n in family.grad_inputs]
                    timed = {
                        "fwd": (
                            functools.partial(family.fwd, xpu, max_len, total_L),
                            family.fwd_bytes(xpu, out),
                        ),
                        "bwd": (
                            functools.partial(
                                torch.autograd.grad,
                                out,
                                grad_targets,
                                grad_out,
                                retain_graph=True,
                            ),
                            family.bwd_bytes(xpu, out),
                        ),
                    }
                    for direction, (fn, num_bytes) in timed.items():
                        seconds, _ = benchmark_torch_function(fn, (), iters=iters)
                        row = {
                            "family": name,
                            "direction": direction,
                            "dtype": str(dtype).removeprefix("torch."),
                            "batch_size": batch_size,
                            "max_len": max_len,
                            "embedding_dim": embedding_dim,
                            "total_lengths": total_L,
                            "time_us": f"{seconds * 1e6:.2f}",
                            "bytes": num_bytes,
                            "gb_per_s": f"{num_bytes / seconds / 1e9:.1f}",
                        }
                        logging.info(", ".join(f"{k}={v}" for k, v in row.items()))
                        rows.append(row)

    # Written only after every shape passed its CPU check.
    with open(output, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    notes = _metadata_path(output)
    notes.write_text(_metadata_markdown(os.path.basename(output), metadata, fields))
    logging.info(f"Wrote {len(rows)} rows to {output} and the run's notes to {notes}")


if __name__ == "__main__":
    # Like the upstream benchmark scripts: an import already installs a root
    # handler, so basicConfig would be a no-op.
    logging.getLogger().setLevel(logging.INFO)
    jagged_sweep(prog_name=f"python -m {_MODULE}")

