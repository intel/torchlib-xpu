# Copyright (c) Meta Platforms, Inc. and affiliates. All rights reserved.
# Copyright (c) 2026 Intel Corporation. All Rights Reserved.
# SPDX-License-Identifier: BSD-3-Clause

import json
import subprocess  # nosec B404
import sys
import unittest
from pathlib import Path

# Operators whose only schema is the fallback in ops_registry.cpp, because
# fbgemm-gpu-cpu does not register them.
XPU_ONLY_OPS = {
    "fbgemm::dense_embedding_nobag_forward_unweighted_xpu",
    "fbgemm::split_embedding_nobag_backward_codegen_dense_unweighted_exact_xpu",
    "fbgemm::split_embedding_nobag_backward_codegen_rowwise_adagrad_unweighted_exact_xpu",
    "fbgemm::split_embedding_nobag_forward_unweighted_xpu",
}

DUMP_SCHEMAS = Path(__file__).with_name("dump_fbgemm_schemas.py")


class TestFallbackSchemas(unittest.TestCase):
    def registered_schemas(self, source: str) -> dict[str, str]:
        # A schema mismatch aborts the process that loads the library, so each
        # registry is read in a separate process.
        result = subprocess.run(  # nosec B603
            [sys.executable, "-I", str(DUMP_SCHEMAS), source],
            capture_output=True,
            text=True,
            timeout=300,
            check=False,
        )
        self.assertEqual(
            result.returncode,
            0,
            f"Reading the {source} schemas failed:\n{result.stderr}",
        )
        return json.loads(result.stdout.splitlines()[-1])

    def test_fallback_schemas_match_fbgemm_gpu(self) -> None:
        fallback = self.registered_schemas("fbgemm_xpu")
        upstream = self.registered_schemas("fbgemm_gpu")

        self.assertTrue(
            upstream.keys() - fallback.keys(),
            "fbgemm_gpu schemas were registered without importing fbgemm_gpu",
        )
        mismatches = [
            f"{name}\n  fallback: {fallback[name]}\n  upstream: {upstream[name]}"
            for name in sorted(fallback.keys() & upstream.keys())
            if fallback[name] != upstream[name]
        ]
        if mismatches:
            self.fail(
                "Fallback schemas differ from fbgemm_gpu:\n" + "\n".join(mismatches)
            )
        self.assertEqual(fallback.keys() - upstream.keys(), XPU_ONLY_OPS)
