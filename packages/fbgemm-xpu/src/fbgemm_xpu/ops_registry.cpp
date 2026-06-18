/*
 * Copyright (c) Meta Platforms, Inc. and affiliates. All rights reserved.
 * Copyright (c) 2026 Intel Corporation. All Rights Reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
 
 #include <Python.h>

#include <ATen/core/Tensor.h>
#include <torch/library.h>

#include "fbgemm_utils/torch_library.h"

using namespace fbgemm_xpu;

extern "C" {
  /**
   * Creates a dummy empty _C module that can be imported from Python.
   *
   * When this module is imported from Python (via 'import fbgemm._C'),
   * it loads the shared library (.so file) and runs all TORCH_LIBRARY
   * static initializers to register the custom operators with PyTorch's
   * dispatch system.
   *
   * @return PyObject* pointer to the created module
   */
  PyObject* PyInit__C(void)
  {
      static struct PyModuleDef module_def = {
          PyModuleDef_HEAD_INIT,
          "_C",   /* name of module - imported as fbgemm._C */
          NULL,   /* module documentation, may be NULL */
          -1,     /* size of per-interpreter state of the module,
                     or -1 if the module keeps state in global variables. */
          NULL,   /* methods - no Python-callable methods needed */
      };
      return PyModule_Create(&module_def);
  }
}
/**
 * Central operator registry for ALL custom operators under the "fbgemm" namespace.
 *
 * Uses TORCH_LIBRARY_FRAGMENT so this can coexist with upstream fbgemm_gpu
 * which may already own the "fbgemm" namespace via TORCH_LIBRARY(fbgemm, m).
 *
 * Operator schemas are declared here; device-specific implementations are
 * registered separately via TORCH_LIBRARY_IMPL(fbgemm, <KEY>, m) in the
 * respective .cpp / .cu files.
 */
TORCH_LIBRARY_FRAGMENT(fbgemm, m)
{
    if (!utils::torch::schemaExists("fbgemm::invert_permute")) {
        m.def(
            "invert_permute(Tensor permute) -> Tensor"
        );
    }
}
