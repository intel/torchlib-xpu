# Copyright (c) 2026 Intel Corporation. All Rights Reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""The TBE training benchmark for the table batched embedding lookups on XPU.

It times the two training lookups registered on XPU, forward and backward, on
the no-bag, unweighted path:

- ``dense``: ``dense_embedding_codegen_lookup_function``, whose backward
  returns the gradient of the whole embedding table;
- ``split``: ``split_embedding_codegen_lookup_rowwise_adagrad_function_pt2``,
  whose backward also applies the rowwise Adagrad update in place.

Workloads are ``TBEDataConfig`` JSON files, loaded with
``TBEDataConfig.from_json``, and their indices and offsets come from upstream's
``generate_requests``. Each workload is checked on CPU before it is timed, so
an incorrect result cannot be recorded as a fast one. The split lookup is
checked against FBGEMM's own CPU kernel for the operator. The dense lookup is
checked against plain torch, because FBGEMM's CPU dense kernel sum-pools even
for no-bag lookups. Run it with::

    python -m fbgemm_xpu.bench.tbe_training \\
        packages/fbgemm-xpu/bench/tbe/configs/*.json --output tbe_training.csv

The operators are called directly, not through
``SplitTableBatchedEmbeddingBagsCodegen``: upstream's
``tbe_training_benchmark.py`` builds that module, which supports CUDA, MTIA and
CPU only. The timings therefore exclude the module's own work, such as index
casts, bounds checks and Python dispatch.
"""

import csv
import dataclasses
import functools
import hashlib
import logging
import os
from collections.abc import Callable
from pathlib import Path

import click
import numpy as np
import torch
from fbgemm_gpu.split_embedding_configs import SparseType
from fbgemm_gpu.split_table_batched_embeddings_ops_common import (
    EmbeddingLocation,
    PoolingMode,
)
from fbgemm_gpu.tbe.bench import TBEDataConfig
from fbgemm_gpu.tbe.bench.tbe_data_config_bench_helper import generate_requests

from fbgemm_xpu.bench.bench_utils import benchmark_torch_function
from fbgemm_xpu.bench.run_notes import notes_markdown, notes_path, run_metadata

_MODULE = "fbgemm_xpu.bench.tbe_training"

# The optimiser and weight initialisation of upstream's tbe_training_benchmark.
LEARNING_RATE = 0.1
EPS = 0.1
WEIGHTS_INIT_BOUND = 0.0003

# The split operator's packed info_B layout, as the training frontend sets it
# for a batch without VBE.
_INFO_B_NUM_BITS = 26
_INFO_B_MASK = (1 << _INFO_B_NUM_BITS) - 1

WEIGHTS_DTYPES: dict[str, SparseType] = {
    "float32": SparseType.FP32,
    "float16": SparseType.FP16,
}
OUTPUT_DTYPE = SparseType.FP32


def _dtype_name(sparse_type: SparseType) -> str:
    return str(sparse_type.as_dtype()).removeprefix("torch.")


def _validate(config: TBEDataConfig, source: str) -> TBEDataConfig:
    """Reject the workloads the XPU lookup kernels cannot run."""
    unsupported = {
        "weighted": config.weighted,
        "mixed_dim": config.mixed_dim,
        "per-table Es or Ds": config.Es is not None or config.Ds is not None,
        "embedding_specs or feature_table_map": config.embedding_specs is not None
        or config.feature_table_map is not None,
        "variable batch sizes (VBE)": config.variable_B()
        or config.batch_params.Bs is not None,
        "variable pooling lengths": config.variable_L()
        or config.pooling_params.Ls is not None,
    }
    found = [name for name, present in unsupported.items() if present]
    if found:
        raise ValueError(
            f"{source}: the XPU lookups support only uniform, unweighted, "
            f"fixed-batch, fixed-length workloads; unsupported: {', '.join(found)}"
        )
    if not config.use_cpu:
        # generate_requests places tensors on CUDA or MTIA unless use_cpu is
        # set; the harness moves them to XPU itself.
        raise ValueError(f"{source}: set use_cpu to true; requests are moved to XPU")
    return config


def load_config(path: str | os.PathLike[str]) -> TBEDataConfig:
    return _validate(TBEDataConfig.from_json(Path(path).read_text()), str(path))


def smoke_config(config: TBEDataConfig) -> TBEDataConfig:
    """A tiny version of ``config`` that exercises every case quickly."""
    return dataclasses.replace(
        config,
        E=min(config.E, 1000),
        batch_params=dataclasses.replace(
            config.batch_params, B=min(config.batch_params.B, 8)
        ),
        pooling_params=dataclasses.replace(
            config.pooling_params, L=min(config.pooling_params.L, 4)
        ),
    )


def describe_config(config: TBEDataConfig) -> str:
    """The parameters of ``config`` that the XPU lookups depend on."""
    ip = config.indices_params
    return (
        f"batch_size={config.batch_params.B}, num_tables={config.T}, "
        f"num_embeddings={config.E}, embedding_dim={config.D}, "
        f"bag_size={config.pooling_params.L}, Zipf q={ip.zipf_q} s={ip.zipf_s}, "
        f"{ip.heavy_hitters.numel()} heavy hitters, "
        f"indices {str(ip.index_dtype).removeprefix('torch.')}, "
        f"offsets {str(ip.offset_dtype).removeprefix('torch.')}"
    )


def upstream_bytes(
    config: TBEDataConfig, weights_dtype: SparseType, output_dtype: SparseType
) -> tuple[int, int]:
    """Forward and backward bytes, as upstream's no-bag formula counts them."""
    lookups = config.batch_params.B * config.T * config.D * config.pooling_params.L
    fwd = int(
        lookups * (weights_dtype.bit_rate() / 8.0 + output_dtype.bit_rate() / 8.0)
    )
    return fwd, 2 * fwd


@dataclasses.dataclass
class Layout:
    """The table layout and the lookup batch, on one device."""

    weights_offsets: torch.Tensor
    D_offsets: torch.Tensor
    total_D: int
    max_D: int
    hash_size_cumsum: torch.Tensor
    total_hash_size_bits: int
    indices: torch.Tensor
    offsets: torch.Tensor

    @classmethod
    def build(
        cls,
        config: TBEDataConfig,
        indices: torch.Tensor,
        offsets: torch.Tensor,
        device: torch.device,
    ) -> "Layout":
        T, E, D = config.T, config.E, config.D
        table = torch.arange(T + 1, dtype=torch.int64)
        return cls(
            weights_offsets=(table[:-1] * E * D).to(device),
            D_offsets=(table * D).to(device=device, dtype=torch.int32),
            total_D=T * D,
            max_D=D,
            hash_size_cumsum=(table * E).to(device),
            total_hash_size_bits=(T * E).bit_length(),
            indices=indices.to(device),
            offsets=offsets.to(device),
        )


LookupCall = Callable[[], tuple[torch.Tensor, list[torch.Tensor]]]


@dataclasses.dataclass(frozen=True)
class Lookup:
    """A TBE training lookup timed by the benchmark.

    ``bind`` takes the layout and the state (``weights``, and for the split
    lookup ``momentum``) and builds every auxiliary tensor once, as the
    training module holds them as buffers, so that the returned call only
    launches the operator. The call returns the output and the inputs to
    differentiate. An empty list means the backward is run for its side
    effect, the in-place optimiser update, and the ``state`` it names is
    compared instead.
    """

    operator: str
    bind: Callable[[Layout, dict[str, torch.Tensor], SparseType], LookupCall]
    state: tuple[str, ...]


def _bind_dense(
    layout: Layout, state: dict[str, torch.Tensor], output_dtype: SparseType
) -> LookupCall:
    weights = state["weights"]
    op = functools.partial(
        torch.ops.fbgemm.dense_embedding_codegen_lookup_function,
        dev_weights=weights,
        weights_offsets=layout.weights_offsets,
        D_offsets=layout.D_offsets,
        total_D=layout.total_D,
        max_D=layout.max_D,
        hash_size_cumsum=layout.hash_size_cumsum,
        total_hash_size_bits=layout.total_hash_size_bits,
        indices=layout.indices,
        offsets=layout.offsets,
        pooling_mode=PoolingMode.NONE.value,
        indice_weights=None,
        feature_requires_grad=None,
        output_dtype=output_dtype.as_int(),
        mixed_D=False,
    )
    return lambda: (op(), [weights])


def _bind_split(
    layout: Layout, state: dict[str, torch.Tensor], output_dtype: SparseType
) -> LookupCall:
    weights, momentum = state["weights"], state["momentum"]
    device = weights.device
    T = layout.weights_offsets.numel()
    row_offsets = layout.hash_size_cumsum[:-1]
    if device.type == "cpu":
        # The CPU kernels take host weights: [host, placements, offsets].
        placements = torch.full((T,), EmbeddingLocation.HOST.value, dtype=torch.int32)
        weights_list = [weights, placements, layout.weights_offsets]
        momentum_list = [momentum, placements, row_offsets]
    else:
        placements = torch.full(
            (T,), EmbeddingLocation.DEVICE.value, device=device, dtype=torch.int32
        )
        weights_list = [
            weights,
            torch.empty(0, device=device, dtype=weights.dtype),
            placements,
            layout.weights_offsets,
            torch.empty((0, layout.max_D), device=device, dtype=weights.dtype),
        ]
        momentum_list = [
            momentum,
            torch.empty(0, device=device, dtype=momentum.dtype),
            placements,
            row_offsets,
        ]
    op = functools.partial(
        torch.ops.fbgemm.split_embedding_codegen_lookup_rowwise_adagrad_function_pt2,
        placeholder_autograd_tensor=torch.zeros(
            (), device=device, dtype=torch.float32, requires_grad=True
        ),
        weights=weights_list,
        D_offsets=layout.D_offsets,
        total_D=layout.total_D,
        max_D=layout.max_D,
        hash_size_cumsum=layout.hash_size_cumsum,
        total_hash_size_bits=layout.total_hash_size_bits,
        indices=layout.indices,
        offsets=layout.offsets,
        pooling_mode=PoolingMode.NONE.value,
        indice_weights=None,
        feature_requires_grad=None,
        output_dtype=output_dtype.as_int(),
        # B_offsets, vbe_output_offsets_feature_rank,
        # vbe_B_offsets_rank_per_feature, lxu_cache_locations,
        # uvm_cache_stats, prev_iter_dev, vbe_output_offsets.
        aux_tensor=[
            None,
            None,
            None,
            torch.empty(0, device=device, dtype=torch.int32),
            torch.empty(0, device=device, dtype=torch.int32),
            None,
            None,
        ],
        # iter, info_B_num_bits, info_B_mask.
        aux_int=[0, _INFO_B_NUM_BITS, _INFO_B_MASK],
        # gwd_lower_bound, max_gradient.
        aux_float=[0.0, 0.0],
        # is_experimental_tbe, use_uniq_cache_locations_bwd,
        # use_homogeneous_placements, apply_global_weight_decay,
        # gradient_clipping, stochastic_rounding, mixed_D.
        aux_bool=[False, False, True, False, False, False, False],
        learning_rate_tensor=torch.tensor(LEARNING_RATE, dtype=torch.float32),
        momentum1=momentum_list,
        # weight_decay_mode.
        optim_int=[0],
        # eps, weight_decay, max_norm.
        optim_float=[EPS, 0.0, 0.0],
    )
    return lambda: (op(), [])


def _backward(
    out: torch.Tensor, grad_out: torch.Tensor, inputs: list[torch.Tensor]
) -> tuple[torch.Tensor, ...]:
    # torch.autograd.grad returns the dense gradient without accumulating it
    # into .grad, which would add a pass over the whole table per iteration.
    if inputs:
        return torch.autograd.grad(out, inputs, grad_out, retain_graph=True)
    torch.autograd.backward(out, grad_out, retain_graph=True)
    return ()


# The output, the gradients and the state after one forward and backward.
LookupResult = tuple[torch.Tensor, list[torch.Tensor], dict[str, torch.Tensor]]


def _run_once(
    lookup: Lookup,
    config: TBEDataConfig,
    indices: torch.Tensor,
    offsets: torch.Tensor,
    initial: dict[str, torch.Tensor],
    grad_out: torch.Tensor,
    device: torch.device,
) -> LookupResult:
    layout = Layout.build(config, indices, offsets, device)
    state = {
        k: v.to(device, copy=True).requires_grad_(not lookup.state)
        for k, v in initial.items()
    }
    out, inputs = lookup.bind(layout, state, OUTPUT_DTYPE)()
    grads = _backward(out, grad_out.to(device), inputs)
    return (
        out.detach().cpu(),
        [g.cpu() for g in grads],
        {k: state[k].detach().cpu() for k in lookup.state},
    )


def _dense_reference(
    config: TBEDataConfig,
    indices: torch.Tensor,
    offsets: torch.Tensor,
    initial: dict[str, torch.Tensor],
    grad_out: torch.Tensor,
) -> LookupResult:
    """The dense lookup in plain torch on CPU.

    FBGEMM's CPU kernel for the dense lookup sum-pools whatever the pooling
    mode, so it cannot check the no-bag path: the forward gathers the rows, and
    the backward scatter-adds the output gradient into the whole table.
    """
    T, E, D, B = config.T, config.E, config.D, config.batch_params.B
    # Bags are table-major: bag b belongs to table b // B.
    bag_table = torch.arange(T).repeat_interleave(B)
    rows = bag_table.repeat_interleave(offsets.diff()) * E + indices.long()
    weights = initial["weights"]
    out = weights.view(T * E, D)[rows].to(OUTPUT_DTYPE.as_dtype())
    grad = torch.zeros(T * E, D).index_add_(0, rows, grad_out.float())
    return out, [grad.to(weights.dtype).flatten()], {}


@dataclasses.dataclass(frozen=True)
class Reference:
    """How a lookup's XPU result is checked on CPU."""

    description: str
    # None runs the operator itself on CPU.
    run: (
        Callable[
            [
                TBEDataConfig,
                torch.Tensor,
                torch.Tensor,
                dict[str, torch.Tensor],
                torch.Tensor,
            ],
            LookupResult,
        ]
        | None
    ) = None


LOOKUPS: dict[str, Lookup] = {
    "dense": Lookup(
        operator="dense_embedding_codegen_lookup_function",
        bind=_bind_dense,
        state=(),
    ),
    "split": Lookup(
        operator="split_embedding_codegen_lookup_rowwise_adagrad_function_pt2",
        bind=_bind_split,
        state=("weights", "momentum"),
    ),
}
REFERENCES: dict[str, Reference] = {
    "dense": Reference(
        "a plain torch gather and index_add on CPU (FBGEMM's CPU dense kernel "
        "sum-pools even for no-bag lookups)",
        _dense_reference,
    ),
    "split": Reference("FBGEMM's CPU kernel for the same operator, host weights"),
}


def _initial_state(
    config: TBEDataConfig, lookup: Lookup, weights_dtype: SparseType, seed: int
) -> dict[str, torch.Tensor]:
    generator = torch.Generator().manual_seed(seed)
    numel = config.T * config.E * config.D
    # In place: an out-of-place expression would hold three copies of a table
    # that reaches 8 GB in the large config.
    weights = (
        torch.rand(numel, generator=generator)
        .mul_(2 * WEIGHTS_INIT_BOUND)
        .sub_(WEIGHTS_INIT_BOUND)
        .to(weights_dtype.as_dtype())
    )
    state = {"weights": weights}
    if "momentum" in lookup.state:
        state["momentum"] = torch.zeros(config.T * config.E, dtype=torch.float32)
    return state


def check_against_cpu(
    name: str,
    op_name: str,
    config: TBEDataConfig,
    indices: torch.Tensor,
    offsets: torch.Tensor,
    initial: dict[str, torch.Tensor],
    grad_out: torch.Tensor,
) -> None:
    """Run one forward and backward on XPU and on CPU and require them to match.

    Both start from the same initial state. The backward check compares the
    weight gradient of the dense lookup and the updated weights and momentum
    of the split lookup.
    """
    lookup, reference = LOOKUPS[op_name], REFERENCES[op_name]
    args = (config, indices, offsets, initial, grad_out)
    if reference.run is None:
        out_cpu, grads_cpu, state_cpu = _run_once(lookup, *args, torch.device("cpu"))
    else:
        out_cpu, grads_cpu, state_cpu = reference.run(*args)
    out_xpu, grads_xpu, state_xpu = _run_once(lookup, *args, torch.device("xpu"))

    _assert_close_chunked(out_xpu, out_cpu, f"{name} fwd")
    if len(grads_xpu) != len(grads_cpu):
        raise AssertionError(f"{name} bwd: {len(grads_xpu)} != {len(grads_cpu)} grads")
    for g_xpu, g_cpu in zip(grads_xpu, grads_cpu):
        _assert_close_chunked(g_xpu, g_cpu, f"{name} bwd")
    for key in lookup.state:
        _assert_close_chunked(state_xpu[key], state_cpu[key], f"{name} bwd {key}")


# Elements compared at once. assert_close makes several temporaries the size
# of its inputs, which for a whole table in the large config would double the
# host memory the check needs.
_CHECK_CHUNK = 1 << 24


def _assert_close_chunked(
    actual: torch.Tensor, expected: torch.Tensor, what: str
) -> None:
    if actual.shape != expected.shape or actual.dtype != expected.dtype:
        # Fails on the metadata before comparing any values.
        torch.testing.assert_close(actual, expected, msg=lambda m: f"{what}: {m}")
    actual, expected = actual.reshape(-1), expected.reshape(-1)
    for start in range(0, actual.numel(), _CHECK_CHUNK):
        end = start + _CHECK_CHUNK
        torch.testing.assert_close(
            actual[start:end],
            expected[start:end],
            msg=lambda m, start=start: f"{what}, from flat element {start}: {m}",
        )


def _time_lookup(
    lookup: Lookup,
    config: TBEDataConfig,
    indices: torch.Tensor,
    offsets: torch.Tensor,
    initial: dict[str, torch.Tensor],
    grad_out: torch.Tensor,
    iters: int,
) -> tuple[float, float]:
    """Median forward and backward seconds on XPU."""
    xpu = torch.device("xpu")
    layout = Layout.build(config, indices, offsets, xpu)
    state = {
        k: v.to(xpu, copy=True).requires_grad_(not lookup.state)
        for k, v in initial.items()
    }
    grad_out = grad_out.to(xpu)
    fwd_s, (out, inputs) = benchmark_torch_function(
        lookup.bind(layout, state, OUTPUT_DTYPE), (), iters=iters
    )
    # The backward reuses one graph; the split lookup's in-place update moves
    # the weights at every iteration, as it does in training. It returns
    # nothing, so each dense gradient of the whole table is freed before the
    # next is allocated rather than two being held at once.
    def backward() -> None:
        _backward(out, grad_out, inputs)

    bwd_s, _ = benchmark_torch_function(backward, (), iters=iters)
    return fwd_s, bwd_s


def _int_list(value: str) -> list[str]:
    return [v.strip() for v in value.split(",") if v.strip()]


def _request(
    config: TBEDataConfig, seed: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """The indices, offsets and output gradient of ``config``, on CPU.

    Seeded, so every precision times the same batch.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    request = generate_requests(config, 1)[0]
    indices, offsets = request.indices.cpu(), request.offsets.cpu()
    grad_out = torch.randn(
        indices.numel(),
        config.D,
        generator=torch.Generator().manual_seed(seed),
        dtype=OUTPUT_DTYPE.as_dtype(),
    )
    return indices, offsets, grad_out


@click.command(name="tbe-training")
@click.argument(
    "configs",
    nargs=-1,
    required=True,
    type=click.Path(exists=True, dir_okay=False),
)
@click.option(
    "--dtypes",
    type=str,
    default=",".join(WEIGHTS_DTYPES),
    help="Comma-separated embedding weight dtypes.",
)
@click.option(
    "--families",
    type=str,
    default=",".join(LOOKUPS),
    help="Comma-separated lookups: dense, split (rowwise Adagrad).",
)
@click.option("--iters", type=int, default=100)
@click.option("--seed", type=int, default=0)
@click.option(
    "--smoke",
    is_flag=True,
    default=False,
    help="Shrinks every config and runs few iterations: exercises every case, "
    "timings are not meaningful.",
)
@click.option("--output", type=click.Path(dir_okay=False), default="tbe_training.csv")
def tbe_training(
    configs: tuple[str, ...],
    dtypes: str,
    families: str,
    iters: int,
    seed: int,
    smoke: bool,
    output: str,
) -> None:
    if not torch.xpu.is_available():
        raise click.UsageError("tbe-training requires an XPU device.")
    try:
        loaded = {Path(path).stem: load_config(path) for path in configs}
        run_dtypes = {d: WEIGHTS_DTYPES[d] for d in _int_list(dtypes)}
        run_lookups = {f: LOOKUPS[f] for f in _int_list(families)}
    except (KeyError, ValueError) as error:
        raise click.UsageError(str(error)) from error
    if smoke:
        loaded = {k: smoke_config(v) for k, v in loaded.items()}
        iters = 5

    metadata = run_metadata(
        _MODULE,
        configs=" ".join(configs),
        config_sha256=", ".join(
            f"{Path(p).name}: {hashlib.sha256(Path(p).read_bytes()).hexdigest()[:16]}"
            for p in configs
        ),
        workloads="; ".join(f"{k}: {describe_config(v)}" for k, v in loaded.items()),
        smoke=smoke,
        cpu_check=", ".join(f"{f}: {REFERENCES[f].description}" for f in run_lookups),
        pooling="none (no-bag), unweighted",
        optimizer=f"split: rowwise Adagrad, learning_rate={LEARNING_RATE}, "
        f"eps={EPS}, no stochastic rounding; dense: none",
        weights_init=f"uniform in [-{WEIGHTS_INIT_BOUND}, {WEIGHTS_INIT_BOUND}]",
        grad_output="standard normal",
        backward="the backward of a graph built once; dense differentiates the "
        "weights with torch.autograd.grad; split runs torch.autograd.backward, "
        "which applies the optimiser update in place",
        dtypes=",".join(run_dtypes),
        output_dtype=_dtype_name(OUTPUT_DTYPE),
        families=", ".join(f"{f} = {lk.operator}" for f, lk in run_lookups.items()),
        iters=iters,
        seed=seed,
    )
    fields = [
        "config",
        "family",
        "direction",
        "dtype",
        "batch_size",
        "num_tables",
        "num_embeddings",
        "embedding_dim",
        "bag_size",
        "total_lengths",
        "time_us",
        "bytes",
        "gb_per_s",
    ]
    rows = []
    for dtype_name, weights_dtype in run_dtypes.items():
        for config_name, config in loaded.items():
            indices, offsets, grad_out = _request(config, seed)
            fwd_bytes, bwd_bytes = upstream_bytes(config, weights_dtype, OUTPUT_DTYPE)
            for family_name, lookup in run_lookups.items():
                name = f"{config_name} {family_name} {dtype_name}"
                initial = _initial_state(config, lookup, weights_dtype, seed)
                check_against_cpu(
                    name, family_name, config, indices, offsets, initial, grad_out
                )
                torch.xpu.empty_cache()
                fwd_s, bwd_s = _time_lookup(
                    lookup, config, indices, offsets, initial, grad_out, iters
                )
                del initial
                torch.xpu.empty_cache()
                for direction, seconds, num_bytes in (
                    ("fwd", fwd_s, fwd_bytes),
                    ("bwd", bwd_s, bwd_bytes),
                ):
                    row = {
                        "config": config_name,
                        "family": family_name,
                        "direction": direction,
                        "dtype": dtype_name,
                        "batch_size": config.batch_params.B,
                        "num_tables": config.T,
                        "num_embeddings": config.E,
                        "embedding_dim": config.D,
                        "bag_size": config.pooling_params.L,
                        "total_lengths": indices.numel(),
                        "time_us": f"{seconds * 1e6:.2f}",
                        "bytes": num_bytes,
                        "gb_per_s": f"{num_bytes / seconds / 1e9:.1f}",
                    }
                    logging.info(", ".join(f"{k}={v}" for k, v in row.items()))
                    rows.append(row)
            del indices, offsets, grad_out

    # Written only after every case passed its CPU check.
    metadata["peak_device_memory_gib"] = (
        f"{torch.xpu.max_memory_allocated() / 2**30:.1f}"
    )
    with open(output, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    notes = notes_path(output)
    notes.write_text(
        notes_markdown(
            os.path.basename(output), "the XPU TBE training lookups", metadata, fields
        )
    )
    logging.info(f"Wrote {len(rows)} rows to {output} and the run's notes to {notes}")


if __name__ == "__main__":
    # An fbgemm_gpu import already installs a root handler, so basicConfig
    # would be a no-op.
    logging.getLogger().setLevel(logging.INFO)
    tbe_training(prog_name=f"python -m {_MODULE}")
