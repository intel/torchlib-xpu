/*
 * Copyright (c) Meta Platforms, Inc. and affiliates. All rights reserved.
 * Copyright (c) 2026 Intel Corporation. All Rights Reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
 
 #include <Python.h>

#include <ATen/core/Tensor.h>
#include <torch/library.h>


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
    m.def("dense_embedding_codegen_lookup_function("
          "    Tensor dev_weights, "
          "    Tensor weights_offsets, "
          "    Tensor D_offsets, "
          "    SymInt total_D, "
          "    SymInt max_D, "
          "    Tensor hash_size_cumsum, "
          "    int total_hash_size_bits, "
          "    Tensor indices, "
          "    Tensor offsets, "
          "    int pooling_mode, "
          "    Tensor? indice_weights, "
          "    Tensor? feature_requires_grad, "
          "    int output_dtype=0, "
          "    Tensor? B_offsets=None, "
          "    Tensor? vbe_output_offsets_feature_rank=None, "
          "    Tensor? vbe_B_offsets_rank_per_feature=None, "
          "    SymInt max_B=-1, "
          "    SymInt max_B_feature_rank=-1, "
          "    SymInt vbe_output_size=-1, "
          "    bool mixed_D=True) -> Tensor");
}
