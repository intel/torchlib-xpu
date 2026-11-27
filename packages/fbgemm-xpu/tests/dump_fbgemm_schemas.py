# Copyright (c) Meta Platforms, Inc. and affiliates. All rights reserved.
# Copyright (c) 2026 Intel Corporation. All Rights Reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""Print the registered fbgemm operator schemas as JSON.

Usage: python -I dump_fbgemm_schemas.py {fbgemm_gpu,fbgemm_xpu}

With fbgemm_gpu, the schemas come from fbgemm_gpu. With fbgemm_xpu, only the
fallback schemas of the fbgemm_xpu native libraries are registered.
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
    package_path = importlib.util.find_spec("fbgemm_xpu").submodule_search_locations
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
