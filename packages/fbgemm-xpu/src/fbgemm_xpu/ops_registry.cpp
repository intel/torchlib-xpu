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


    if (!utils::torch::schemaExists("fbgemm::permute_1D_sparse_data")) {
        m.def(
            "permute_1D_sparse_data("
            "    Tensor permute, "
            "    Tensor lengths, "
            "    Tensor values, "
            "    Tensor? weights=None, "
            "    SymInt? permuted_lengths_sum=None"
            ") -> (Tensor, Tensor, Tensor?)"
        );
    }

    if (!utils::torch::schemaExists("fbgemm::asynchronous_complete_cumsum")) {
        m.def(
            "asynchronous_complete_cumsum(Tensor t_in) -> Tensor"
        );
    }

    if (!utils::torch::schemaExists("fbgemm::asynchronous_exclusive_cumsum")) {
        m.def(
            "asynchronous_exclusive_cumsum(Tensor t_in) -> Tensor"
        );
    }

    if (!utils::torch::schemaExists("fbgemm::asynchronous_inclusive_cumsum")) {
        m.def(
            "asynchronous_inclusive_cumsum(Tensor t_in) -> Tensor"
        );
    }

    if (!utils::torch::schemaExists("fbgemm::permute_2D_sparse_data")) {
        m.def(
            "permute_2D_sparse_data("
            "    Tensor permute, "
            "    Tensor lengths, "
            "    Tensor values, "
            "    Tensor? weights=None, "
            "    SymInt? permuted_lengths_sum=None"
            ") -> (Tensor, Tensor, Tensor?)"
        );
    }

    if (!utils::torch::schemaExists("fbgemm::permute_2D_sparse_preallocated_out")) {
        m.def(
            "permute_2D_sparse_preallocated_out("
            "    Tensor permute, "
            "    Tensor lengths, "
            "    Tensor values, "
            "    Tensor? weights=None, "
            "    SymInt? permuted_lengths_sum=None, "
            "    Tensor? permuted_lengths_out=None, "
            "    Tensor? permuted_indices_out=None, "
            "    Tensor? permuted_weights_out=None"
            ") -> (Tensor, Tensor, Tensor?)"
        );
    }

    if (!utils::torch::schemaExists("fbgemm::get_infos_metadata")) {
        m.def(
            "get_infos_metadata(Tensor unused, int B, int T) -> (int, int)"
        );
    }

    if (!utils::torch::schemaExists("fbgemm::block_bucketize_sparse_features")) {
        m.def(
            "block_bucketize_sparse_features("
            "    Tensor lengths, "
            "    Tensor indices, "
            "    bool bucketize_pos, "
            "    bool sequence, "
            "    Tensor block_sizes, "
            "    int my_size, "
            "    Tensor? weights=None, "
            "    Tensor? batch_size_per_feature=None, "
            "    int max_B=0, "
            "    Tensor[]? block_bucketize_pos=None, "
            "    bool keep_orig_idx=False, "
            "    Tensor? total_num_blocks=None, "
            "    Tensor? keep_orig_idx_per_feature=None"
            ") -> (Tensor, Tensor, Tensor?, Tensor?, Tensor?)"
        );
    }

    if (!utils::torch::schemaExists("fbgemm::block_bucketize_sparse_features_inference")) {
        m.def(
            "block_bucketize_sparse_features_inference("
            "    Tensor lengths, "
            "    Tensor indices, "
            "    bool bucketize_pos, "
            "    bool sequence, "
            "    Tensor block_sizes, "
            "    int my_size, "
            "    Tensor? weights=None, "
            "    Tensor? batch_size_per_feature=None, "
            "    int max_B=0, "
            "    Tensor[]? block_bucketize_pos=None, "
            "    bool return_bucket_mapping=False, "
            "    bool keep_orig_idx=False, "
            "    Tensor? total_num_blocks=None, "
            "    Tensor? keep_orig_idx_per_feature=None"
            ") -> (Tensor, Tensor, Tensor?, Tensor?, Tensor?, Tensor?)"
        );
    }

    if (!utils::torch::schemaExists("fbgemm::populate_bucketized_permute")) {
        m.def(
            "populate_bucketized_permute("
            "    Tensor lengths, "
            "    Tensor bucketized_lengths, "
            "    Tensor bucket_mapping"
            ") -> Tensor"
        );
    }
}
