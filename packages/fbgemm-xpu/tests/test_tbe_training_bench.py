# Copyright (c) 2026 Intel Corporation. All Rights Reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for the TBE training benchmark of the XPU lookup operators."""

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
    assert [p.stem for p in _CONFIGS] == ["large", "medium", "small"]  # nosec B101


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
    lookups = 2048 * 8 * 20 * 128
    assert tbe_training.upstream_bytes(  # nosec B101
        config, SparseType.FP32, SparseType.FP32
    ) == (lookups * 8, lookups * 16)
    assert tbe_training.upstream_bytes(  # nosec B101
        config, SparseType.FP16, SparseType.FP32
    ) == (lookups * 6, lookups * 12)


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


def test_variable_pooling_length_is_rejected(small_config_path):
    config = tbe_training.load_config(small_config_path)
    config = dataclasses.replace(
        config,
        pooling_params=dataclasses.replace(config.pooling_params, sigma_L=4),
    )
    with pytest.raises(ValueError, match="variable pooling lengths"):
        tbe_training._validate(config, "test")


def test_smoke_config_shrinks_only_sizes(small_config_path):
    config = tbe_training.load_config(small_config_path)
    smoke = tbe_training.smoke_config(config)
    assert (smoke.E, smoke.batch_params.B, smoke.pooling_params.L) == (  # nosec B101
        1000,
        8,
        4,
    )
    assert (smoke.T, smoke.D) == (config.T, config.D)  # nosec B101


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
