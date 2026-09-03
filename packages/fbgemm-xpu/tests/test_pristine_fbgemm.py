# Copyright (c) Meta Platforms, Inc. and affiliates. All rights reserved.
# Copyright (c) 2026 Intel Corporation. All Rights Reserved.
# SPDX-License-Identifier: BSD-3-Clause

import subprocess  # nosec B404
import sys
import sysconfig
import tempfile
import textwrap
import unittest
from importlib.metadata import version
from pathlib import Path

import fbgemm_gpu
import fbgemm_xpu  # noqa: F401
import torch


class TestPristineFbgemm(unittest.TestCase):
    def assert_existing_xpu_operator_works(self) -> None:
        values = torch.tensor([2, 0, 3], device="xpu", dtype=torch.int64)
        actual = torch.ops.fbgemm.asynchronous_complete_cumsum(values)

        torch.testing.assert_close(
            actual.cpu(),
            torch.tensor([0, 2, 2, 5], dtype=torch.int64),
        )

    def test_imports_pinned_fbgemm_from_environment(self) -> None:
        package_path = Path(fbgemm_gpu.__file__).resolve()
        environment_packages = Path(sysconfig.get_path("purelib")).resolve()

        self.assertTrue(
            package_path.is_relative_to(environment_packages),
            f"{package_path} is not installed under {environment_packages}",
        )
        self.assertEqual(version("fbgemm-gpu-cpu"), "1.8.0")

    @unittest.skipUnless(torch.xpu.is_available(), "XPU is required")
    def test_existing_xpu_operator(self) -> None:
        self.assert_existing_xpu_operator_works()

    @unittest.skipUnless(torch.xpu.is_available(), "XPU is required")
    def test_pristine_frontend_failure_is_process_safe(self) -> None:
        script = textwrap.dedent(
            """
            import fbgemm_xpu

            from fbgemm_gpu.split_embedding_configs import EmbOptimType
            from fbgemm_gpu.split_table_batched_embeddings_ops_common import (
                EmbeddingLocation,
            )
            from fbgemm_gpu.split_table_batched_embeddings_ops_training import (
                SplitTableBatchedEmbeddingBagsCodegen,
            )
            from fbgemm_gpu.tbe.config.embedding_config import ComputeDevice

            print("FRONTEND_IMPORTS_OK", flush=True)
            SplitTableBatchedEmbeddingBagsCodegen(
                [
                    (
                        4,
                        4,
                        EmbeddingLocation.DEVICE,
                        ComputeDevice.XPU,
                    )
                ],
                optimizer=EmbOptimType.EXACT_SGD,
            )
            """
        )

        with tempfile.TemporaryDirectory() as working_directory:
            try:
                result = subprocess.run(  # nosec B603
                    [sys.executable, "-c", script],
                    cwd=working_directory,
                    capture_output=True,
                    text=True,
                    timeout=300,
                    check=False,
                )
            except subprocess.TimeoutExpired as error:
                self.fail(f"Pristine FBGEMM frontend hung: {error}")

        self.assertGreater(
            result.returncode,
            0,
            f"Frontend process terminated abnormally: {result.stderr}",
        )
        self.assertIn("FRONTEND_IMPORTS_OK", result.stdout)
        self.assertRegex(
            result.stderr,
            r"AttributeError: (type object 'ComputeDevice' has no attribute 'XPU'|XPU)",
        )

        self.assert_existing_xpu_operator_works()
