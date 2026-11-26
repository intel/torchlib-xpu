# Copyright (c) Meta Platforms, Inc. and affiliates. All rights reserved.
# Copyright (c) 2026 Intel Corporation. All Rights Reserved.
# SPDX-License-Identifier: BSD-3-Clause

import functools
import json
import subprocess  # nosec B404
import sys
import tempfile
import textwrap
import unittest

# Operators whose only schema is the fallback in ops_registry.cpp, because
# fbgemm-gpu-cpu does not register them.
XPU_ONLY_OPS = {
    "fbgemm::dense_embedding_nobag_forward_unweighted_xpu",
    "fbgemm::split_embedding_nobag_backward_codegen_dense_unweighted_exact_xpu",
    "fbgemm::split_embedding_nobag_backward_codegen_rowwise_adagrad_unweighted_exact_xpu",
    "fbgemm::split_embedding_nobag_forward_unweighted_xpu",
}

DUMP_SCHEMAS = textwrap.dedent(
    """
    import importlib.machinery
    import importlib.util
    import json
    import sys

    import torch

    if sys.argv[1] == "fbgemm_gpu":
        import fbgemm_gpu  # noqa: F401
    else:
        # fbgemm_xpu/__init__.py imports fbgemm_gpu first, so load the native
        # libraries directly to register only the fallback schemas.
        package_path = importlib.util.find_spec(
            "fbgemm_xpu"
        ).submodule_search_locations
        for name in ("_C", "_C_training"):
            spec = importlib.machinery.PathFinder.find_spec(name, package_path)
            torch.ops.load_library(spec.origin)
        if "fbgemm_gpu" in sys.modules:
            sys.exit("fbgemm_gpu was imported")

    schemas = {}
    for schema in torch._C._jit_get_all_schemas():
        if schema.name.startswith("fbgemm::"):
            name = schema.name
            if schema.overload_name:
                name += "." + schema.overload_name
            schemas[name] = str(schema)
    print(json.dumps(schemas))
    """
)


@functools.cache
def dump_schemas(source: str) -> subprocess.CompletedProcess:
    # A schema mismatch aborts the process that loads the library, so each
    # registry is read in a separate process.
    with tempfile.TemporaryDirectory() as working_directory:
        return subprocess.run(  # nosec B603
            [sys.executable, "-c", DUMP_SCHEMAS, source],
            cwd=working_directory,
            capture_output=True,
            text=True,
            timeout=300,
            check=False,
        )


class TestFallbackSchemas(unittest.TestCase):
    def registered_schemas(self, source: str) -> dict[str, str]:
        result = dump_schemas(source)
        self.assertEqual(
            result.returncode,
            0,
            f"Reading the {source} schemas failed:\n{result.stderr}",
        )
        return json.loads(result.stdout.splitlines()[-1])

    def test_extensions_load_without_fbgemm_gpu(self) -> None:
        self.registered_schemas("fbgemm_xpu")

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
