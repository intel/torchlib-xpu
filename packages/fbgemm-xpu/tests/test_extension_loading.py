# Copyright (c) Meta Platforms, Inc. and affiliates. All rights reserved.
# Copyright (c) 2026 Intel Corporation. All Rights Reserved.
# SPDX-License-Identifier: BSD-3-Clause

import importlib
import importlib.abc
import sys
import unittest

import fbgemm_xpu
import torch


class BlockedExtensionFinder(importlib.abc.MetaPathFinder):
    def __init__(self, blocked_module: str) -> None:
        self.blocked_module = blocked_module

    def find_spec(self, fullname, path=None, target=None):
        if fullname == self.blocked_module:
            raise ImportError(f"blocked native extension: {fullname}")
        return None


class TestExtensionLoading(unittest.TestCase):
    def test_training_extension_registers_autograd_xpu(self) -> None:
        operators = (
            "fbgemm::dense_embedding_codegen_lookup_function",
            "fbgemm::split_embedding_codegen_lookup_rowwise_adagrad_function_pt2",
        )

        for operator in operators:
            with self.subTest(operator=operator):
                self.assertTrue(
                    torch._C._dispatch_has_kernel_for_dispatch_key(
                        operator, "AutogradXPU"
                    )
                )

    def test_native_extensions_load_in_schema_order(self) -> None:
        loaded_modules = list(sys.modules)

        self.assertIs(fbgemm_xpu._C, sys.modules["fbgemm_xpu._C"])
        self.assertIs(
            fbgemm_xpu._C_training, sys.modules["fbgemm_xpu._C_training"]
        )
        self.assertLess(
            loaded_modules.index("fbgemm_xpu._C"),
            loaded_modules.index("fbgemm_xpu._C_training"),
        )

    def test_reimport_does_not_reload_native_extensions(self) -> None:
        core_extension = fbgemm_xpu._C
        training_extension = fbgemm_xpu._C_training

        reloaded_package = importlib.reload(fbgemm_xpu)

        self.assertIs(reloaded_package._C, core_extension)
        self.assertIs(reloaded_package._C_training, training_extension)

    def test_missing_native_extension_is_not_silenced(self) -> None:
        for blocked_module in ("fbgemm_xpu._C", "fbgemm_xpu._C_training"):
            with self.subTest(blocked_module=blocked_module):
                package = sys.modules.pop("fbgemm_xpu")
                extension = sys.modules.pop(blocked_module)
                finder = BlockedExtensionFinder(blocked_module)
                sys.meta_path.insert(0, finder)

                try:
                    with self.assertRaisesRegex(
                        ImportError, f"blocked native extension: {blocked_module}"
                    ):
                        importlib.import_module("fbgemm_xpu")
                finally:
                    sys.meta_path.remove(finder)
                    sys.modules.pop("fbgemm_xpu", None)
                    sys.modules["fbgemm_xpu"] = package
                    sys.modules[blocked_module] = extension
