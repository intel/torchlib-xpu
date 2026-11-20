# Copyright (c) 2026 Intel Corporation. All Rights Reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for the FBGEMM_<FLAG> feature gate environment variables."""

import os
import subprocess  # nosec B404
import sys

import pytest
import torch

pytestmark = pytest.mark.skipif(
    not torch.xpu.is_available(), reason="requires an XPU device"
)

# A one-row dense lookup. Its first call reads the
# FBGEMM_TBE_ANNOTATE_KINETO_TRACE gate.
_DENSE_LOOKUP = """
import fbgemm_xpu  # noqa: F401
import torch

d = 4
weights = torch.arange(1, d + 1, device="xpu", dtype=torch.float32)
output = torch.ops.fbgemm.dense_embedding_codegen_lookup_function(
    weights,
    torch.tensor([0], device="xpu", dtype=torch.int64),
    torch.tensor([0, d], device="xpu", dtype=torch.int32),
    d,
    d,
    torch.tensor([0, 1], device="xpu", dtype=torch.int64),
    1,
    torch.tensor([0], device="xpu", dtype=torch.int32),
    torch.tensor([0, 1], device="xpu", dtype=torch.int32),
    2,
    None,
    None,
    0,
    None,
    None,
    None,
    -1,
    -1,
    -1,
    False,
)
torch.testing.assert_close(output.cpu(), weights.cpu().unsqueeze(0))
"""


@pytest.mark.parametrize("value", ["99999999999", "not-a-number"])
def test_unparsable_gate_value_disables_the_gate(value):
    # The gate is read once per process, so each value needs a fresh process.
    env = dict(os.environ, FBGEMM_TBE_ANNOTATE_KINETO_TRACE=value)
    # Runs a fixed snippet with sys.executable.
    result = subprocess.run(  # nosec B603
        [sys.executable, "-c", _DENSE_LOOKUP],
        env=env,
        capture_output=True,
        text=True,
        timeout=600,
    )
    assert result.returncode == 0, result.stderr  # nosec B101
