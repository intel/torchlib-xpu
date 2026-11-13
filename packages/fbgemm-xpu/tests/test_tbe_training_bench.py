# Copyright (c) 2026 Intel Corporation. All Rights Reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for the TBE training benchmark of the XPU lookup operators."""

import csv
import dataclasses
import json
from pathlib import Path

import pytest
import torch
from click.testing import CliRunner
from fbgemm_gpu.split_embedding_configs import SparseType
from fbgemm_gpu.tbe.bench import TBEDataConfig
from fbgemm_xpu.bench import tbe_training

_CONFIGS = sorted(
    (Path(__file__).resolve().parents[1] / "bench" / "tbe" / "configs").glob("*.json")
)

requires_xpu = pytest.mark.skipif(
    not torch.xpu.is_available(), reason="requires an XPU device"
)


@pytest.fixture
def small_config_path():
    return next(p for p in _CONFIGS if p.stem == "small")


def test_committed_configs_exist():
    assert [p.stem for p in _CONFIGS] == [  # nosec B101
        "large",
        "medium",
        "skewed",
        "small",
    ]


def test_skewed_config_reaches_the_repeated_row_backward_paths():
    # The XPU backward has a CTA-per-row path from 32 lookups of a row and
    # several work-groups with atomics above 1,024; near-uniform indices
    # repeat a row at most 11 times in the other configs.
    config = tbe_training.load_config(next(p for p in _CONFIGS if p.stem == "skewed"))
    indices, offsets, _ = tbe_training._request(config, 0)
    counts = torch.bincount(tbe_training._table_rows(config, indices, offsets))
    assert ((counts >= 32) & (counts <= 1024)).sum() > 100  # nosec B101
    assert (counts > 1024).sum() > 10  # nosec B101


@pytest.mark.parametrize("path", _CONFIGS, ids=lambda p: p.stem)
def test_committed_config_is_complete_and_canonical(path):
    # Every TBEDataConfig field is present, so from_json warns about nothing,
    # and the file is what TBEDataConfig writes for itself.
    text = path.read_text()
    fields = {f.name for f in dataclasses.fields(TBEDataConfig)}
    assert set(json.loads(text)) == fields  # nosec B101
    config = tbe_training.load_config(path)
    assert config.json(format=True) + "\n" == text  # nosec B101
    assert config.D == 128 and not config.weighted  # nosec B101


def test_upstream_bytes_match_upstream_nobag_formula():
    config = tbe_training.load_config(
        next(p for p in _CONFIGS if p.stem == "medium")
    )
    # With a fixed L, counting the indices is upstream's B x T x L.
    num_indices = 2048 * 8 * 20
    lookups = num_indices * 128
    assert tbe_training.upstream_bytes(  # nosec B101
        config, num_indices, SparseType.FP32, SparseType.FP32
    ) == (lookups * 8, lookups * 16)
    assert tbe_training.upstream_bytes(  # nosec B101
        config, num_indices, SparseType.FP16, SparseType.FP32
    ) == (lookups * 6, lookups * 12)
    # With variable bag lengths, the indices actually drawn are counted.
    assert tbe_training.upstream_bytes(  # nosec B101
        config, 1000, SparseType.FP32, SparseType.FP32
    ) == (1000 * 128 * 8, 1000 * 128 * 16)


@pytest.mark.parametrize(
    "change, reason",
    [
        ({"weighted": True}, "weighted"),
        ({"mixed_dim": True}, "mixed_dim"),
        ({"use_cpu": False}, "use_cpu"),
    ],
)
def test_unsupported_configs_are_rejected(small_config_path, change, reason):
    config = dataclasses.replace(tbe_training.load_config(small_config_path), **change)
    with pytest.raises(ValueError, match=reason):
        tbe_training._validate(config, "test")


# The two shapes the XPU lookups were checked on: lengths drawn around L, and
# DLRM-v3's mix of a jagged history and single-ID features.
_VARIABLE_LENGTHS = {"sigma_L": {"sigma_L": 10}, "Ls": {"Ls": [50, 1, 1, 20]}}


def _with_pooling(config, **pooling):
    return dataclasses.replace(
        config,
        pooling_params=dataclasses.replace(config.pooling_params, **pooling),
    )


@pytest.mark.parametrize("pooling", _VARIABLE_LENGTHS.values(), ids=_VARIABLE_LENGTHS)
def test_variable_bag_lengths_are_accepted(small_config_path, pooling):
    config = _with_pooling(tbe_training.load_config(small_config_path), **pooling)
    assert tbe_training._validate(config, "test") is config  # nosec B101


def test_smoke_config_shrinks_only_sizes(small_config_path):
    config = tbe_training.load_config(small_config_path)
    smoke = tbe_training.smoke_config(config)
    assert (smoke.E, smoke.batch_params.B, smoke.pooling_params.L) == (  # nosec B101
        1000,
        8,
        4,
    )
    assert (smoke.T, smoke.D) == (config.T, config.D)  # nosec B101


def test_dense_reference_accumulates_in_float64(small_config_path):
    # One row read 1,001 times: a float32 sum in order would lose every 1
    # added to 2**24, while float64 gives 2**24 + 1000, which float32 holds.
    config = tbe_training.load_config(small_config_path)
    config = dataclasses.replace(
        config,
        T=1,
        E=1,
        D=1,
        batch_params=dataclasses.replace(config.batch_params, B=1),
        pooling_params=dataclasses.replace(config.pooling_params, L=1001),
    )
    indices = torch.zeros(1001, dtype=torch.int64)
    offsets = torch.tensor([0, 1001])
    grad_out = torch.ones(1001, 1)
    grad_out[0] = 2**24
    initial = {"weights": torch.zeros(1)}
    _, (grad,), _ = tbe_training._dense_reference(
        config, indices, offsets, initial, grad_out
    )
    assert grad.item() == 2**24 + 1000  # nosec B101


def test_chunked_check_finds_mismatch_past_first_chunk(monkeypatch):
    monkeypatch.setattr(tbe_training, "_CHECK_CHUNK", 4)
    expected = torch.zeros(3, 4)
    tbe_training._assert_close_chunked(expected.clone(), expected, "same")
    actual = expected.clone()
    actual[2, 1] = 1
    with pytest.raises(AssertionError, match="bad, from flat element 8"):
        tbe_training._assert_close_chunked(actual, expected, "bad")
    with pytest.raises(AssertionError, match="shape"):
        tbe_training._assert_close_chunked(expected.reshape(4, 3), expected, "shape")


def test_device_copy_in_slices_matches_whole_copy(monkeypatch):
    # 10 bytes: slices of 5 FP16 elements, the last one shorter.
    monkeypatch.setattr(tbe_training, "_COPY_BYTES", 10)
    source = torch.arange(12, dtype=torch.float16)
    copy = tbe_training._to_device(source, torch.device("cpu"))
    assert copy.data_ptr() != source.data_ptr()  # nosec B101
    torch.testing.assert_close(copy, source)


def _run(args, tmp_path):
    output = tmp_path / "tbe.csv"
    result = CliRunner().invoke(
        tbe_training.tbe_training, [*args, "--output", str(output)]
    )
    return result, output


@requires_xpu
def test_smoke_run_writes_rows_and_notes(small_config_path, tmp_path):
    result, output = _run(["--smoke", str(small_config_path)], tmp_path)
    assert result.exit_code == 0, result.output  # nosec B101

    rows = output.read_text().splitlines()
    assert rows[0] == (  # nosec B101
        "config,family,direction,dtype,batch_size,num_tables,num_embeddings,"
        "embedding_dim,bag_size,total_lengths,time_us,bytes,gb_per_s"
    )
    # Then float32 and float16 x dense and split x fwd and bwd, dtype first,
    # as jagged-sweep orders its rows.
    assert len(rows) == 1 + 8  # nosec B101
    assert [r.split(",")[:4] for r in rows[1:3]] == [  # nosec B101
        ["small", "dense", "fwd", "float32"],
        ["small", "dense", "bwd", "float32"],
    ]
    assert rows[-1].split(",")[:4] == ["small", "split", "bwd", "float16"]  # nosec B101
    notes = output.with_suffix(".md").read_text()
    # Like jagged-sweep's notes: the column list, then the key-value table.
    assert notes.splitlines()[4:6] == ["", "| Key | Value |"]  # nosec B101
    assert "| dtypes | float32,float16 |" in notes  # nosec B101
    assert "| families | dense = " in notes  # nosec B101
    # The notes describe the shrunk workload that actually ran.
    assert (  # nosec B101
        "small: batch_size=8, num_tables=4, num_embeddings=1000, "
        "embedding_dim=128, bag_size=4," in notes
    )


@requires_xpu
def test_timing_process_times_the_checked_batch(small_config_path):
    # The run checks a batch in its own process and times one regenerated
    # from the seed in another, so the two must be the same batch.
    config = tbe_training.smoke_config(tbe_training.load_config(small_config_path))
    indices, offsets, _ = tbe_training._request(config, 0)
    timing = tbe_training._time_in_fresh_process(config, "dense", SparseType.FP32, 0, 2)
    assert timing.request_digest == tbe_training._request_digest(  # nosec B101
        indices, offsets
    )
    assert timing.fwd_s > 0 and timing.bwd_s > 0  # nosec B101
    assert timing.peak_device_bytes > 0  # nosec B101


@requires_xpu
def test_smoke_run_checks_variable_bag_lengths(small_config_path, tmp_path):
    small = tbe_training.load_config(small_config_path)
    paths = []
    for name, pooling in _VARIABLE_LENGTHS.items():
        path = tmp_path / f"{name}.json"
        path.write_text(_with_pooling(small, **pooling).json(format=True))
        paths.append(str(path))
    # Every case passes its CPU check before a row is written.
    result, output = _run(["--smoke", "--dtypes", "float32", *paths], tmp_path)
    assert result.exit_code == 0, result.output  # nosec B101

    with open(output) as f:
        rows = {(r["config"], r["family"], r["direction"]): r for r in csv.DictReader(f)}
    assert len(rows) == 8  # nosec B101
    # Smoke caps the lengths at 4: Ls [4, 1, 1, 4] for 8 bags per table.
    ls_row = rows["Ls", "dense", "fwd"]
    assert (ls_row["total_lengths"], ls_row["bag_size"]) == ("80", "2.5")  # nosec B101
    assert int(ls_row["bytes"]) == 80 * 128 * 8  # nosec B101
    notes = output.with_suffix(".md").read_text()
    assert "bag sizes per table [4, 1, 1, 4]" in notes  # nosec B101
    assert "bag_size normal mean 4 sigma 10" in notes  # nosec B101


@requires_xpu
def test_cpu_mismatch_aborts_without_writing(small_config_path, tmp_path, monkeypatch):
    reference = tbe_training.REFERENCES["dense"]

    def wrong(*args):
        out, grads, state = reference.run(*args)
        return out + 1, grads, state

    monkeypatch.setitem(
        tbe_training.REFERENCES,
        "dense",
        dataclasses.replace(reference, run=wrong),
    )
    result, output = _run(
        ["--smoke", "--families", "dense", str(small_config_path)], tmp_path
    )

    assert result.exit_code != 0  # nosec B101
    assert "fwd" in str(result.exception)  # nosec B101
    assert not output.exists()  # nosec B101
    assert not output.with_suffix(".md").exists()  # nosec B101
