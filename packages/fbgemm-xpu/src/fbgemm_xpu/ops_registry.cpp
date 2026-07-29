/*
 * Copyright (c) Meta Platforms, Inc. and affiliates. All rights reserved.
 * Copyright (c) 2026 Intel Corporation. All Rights Reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */

#include <Python.h>
#include <ATen/core/Tensor.h>
#include <torch/library.h>

#include "fbgemm_utils/torch_library.h"

using fbgemm_xpu::utils::torch::schemaExists;

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
PyObject* PyInit__C(void) {
    static struct PyModuleDef module_def = {
        PyModuleDef_HEAD_INIT,
        "_C",  // name of module - imported as fbgemm._C
        NULL,  // module documentation, may be NULL
        -1,    // size of per-interpreter state of the module,
               //    or -1 if the module keeps state in global variables.
        NULL,  // methods - no Python-callable methods needed
    };
    return PyModule_Create(&module_def);
}
}
/**
 * Central operator registry for ALL custom operators under the "fbgemm"
 * namespace.
 *
 * Uses TORCH_LIBRARY_FRAGMENT so this can coexist with upstream fbgemm_gpu
 * which may already own the "fbgemm" namespace via TORCH_LIBRARY(fbgemm, m).
 *
 * Operator schemas are declared here; device-specific implementations are
 * registered separately via TORCH_LIBRARY_IMPL(fbgemm, <KEY>, m) in the
 * respective .cpp / .cu files.
 */
TORCH_LIBRARY_FRAGMENT(fbgemm, m) {
    // ===== Dense Embedding Operators =====

    if (!schemaExists(
            "fbgemm::dense_embedding_codegen_lookup_function")) {
        m.def(
            "dense_embedding_codegen_lookup_function("
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

    if (!schemaExists(
            "fbgemm::dense_embedding_nobag_forward_unweighted_xpu")) {
        m.def(
            "dense_embedding_nobag_forward_unweighted_xpu("
            "    Tensor dev_weights, "
            "    Tensor weights_offsets, "
            "    SymInt D, "
            "    Tensor indices, "
            "    Tensor offsets, "
            "    int output_dtype, "
            "    bool is_experimental"
            ") -> Tensor");
    }

    // ===== Split Embedding Operators =====

    if (!schemaExists("fbgemm::split_embedding_nobag_backward_"
                                    "codegen_dense_unweighted_exact_xpu")) {
        m.def(
            "split_embedding_nobag_backward_codegen_dense_unweighted_exact_xpu("
            "    Tensor grad_output, "
            "    Tensor(a!) dev_weights, "
            "    Tensor weights_offsets, "
            "    SymInt D, "
            "    Tensor hash_size_cumsum, "
            "    int total_hash_size_bits, "
            "    Tensor indices, "
            "    Tensor offsets, "
            "    int unused_, "
            "    int max_segment_length_per_warp, "
            "    float unused = 0) -> Tensor");
    }

    if (!schemaExists("fbgemm::split_embedding_codegen_lookup_"
                                    "rowwise_adagrad_function_pt2")) {
        m.def(
            "split_embedding_codegen_lookup_rowwise_adagrad_function_pt2("
            "    Tensor placeholder_autograd_tensor, "
            "    Tensor[](a!) weights, "
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
            "    int output_dtype, "
            "    Tensor?[](e!) aux_tensor, "
            "    int[] aux_int, "
            "    float[] aux_float, "
            "    bool[] aux_bool, "
            "    Tensor[](g!) momentum1, "
            "    Tensor learning_rate_tensor, "
            "    int[] optim_int, "
            "    float[] optim_float, "
            "    SymInt max_B=-1, "
            "    SymInt max_B_feature_rank=-1, "
            "    SymInt vbe_output_size=-1, "
            "    Tensor? vbe_output=None "
            ") -> Tensor");
    }

    if (!schemaExists("fbgemm::split_embedding_nobag_codegen_"
                                    "forward_unweighted_pt2_wrapper")) {
        m.def(
            "split_embedding_nobag_codegen_forward_unweighted_pt2_wrapper("
            "    Tensor host_weights, "
            "    Tensor dev_weights, "
            "    Tensor uvm_weights, "
            "    Tensor lxu_cache_weights, "
            "    Tensor weights_placements, "
            "    Tensor weights_offsets, "
            "    SymInt D, "
            "    Tensor hash_size_cumsum, "
            "    Tensor indices, "
            "    Tensor offsets, "
            "    Tensor lxu_cache_locations, "
            "    Tensor(f!) uvm_cache_stats, "
            "    bool is_experimental, "
            "    int output_dtype "
            ") -> Tensor");
    }

    if (!schemaExists(
            "fbgemm::split_embedding_nobag_backward_codegen_rowwise_adagrad_"
            "unweighted_exact_xpu")) {
        m.def(
            "split_embedding_nobag_backward_codegen_rowwise_adagrad_unweighted_"
            "exact_xpu("
            "    Tensor grad_output, "
            "    Tensor(a!) dev_weights, "
            "    Tensor(b!) uvm_weights, "
            "    Tensor lxu_cache_weights, "
            "    Tensor weights_placements, "
            "    Tensor weights_offsets, "
            "    SymInt D, "
            "    Tensor hash_size_cumsum, "
            "    int total_hash_size_bits, "
            "    Tensor indices, "
            "    Tensor offsets, "
            "    Tensor lxu_cache_locations, "
            "    int unused_, "
            "    int max_segment_length_per_warp, "
            "    bool stochastic_rounding, "
            "    int info_B_num_bits, "
            "    int info_B_mask_int64, "
            "    bool use_uniq_cache_locations, "
            "    bool use_homogeneous_placements, "
            "    Tensor(h!) momentum1_dev, "
            "    Tensor(i!) momentum1_uvm, "
            "    Tensor momentum1_placements, "
            "    Tensor momentum1_offsets, "
            "    Tensor learning_rate_tensor, "
            "    float eps = 0, "
            "    float weight_decay = 0.0, "
            "    int weight_decay_mode = 0, "
            "    float max_norm = 0.0"
            ") -> Tensor");
    }

    if (!schemaExists(
            "fbgemm::split_embedding_nobag_forward_unweighted_xpu")) {
        m.def(
            "split_embedding_nobag_forward_unweighted_xpu("
            "    Tensor dev_weights, "
            "    Tensor uvm_weights, "
            "    Tensor lxu_cache_weights, "
            "    Tensor weights_placements, "
            "    Tensor weights_offsets, "
            "    SymInt D, "
            "    Tensor indices, "
            "    Tensor offsets, "
            "    Tensor lxu_cache_locations, "
            "    Tensor uvm_cache_stats, "
            "    int output_dtype, "
            "    bool is_experimental"
            ") -> Tensor");
    }

    if (!schemaExists(
            "fbgemm::split_embedding_nobag_backward_codegen_rowwise_adagrad_"
            "unweighted_pt2_wrapper")) {
        m.def(
            "split_embedding_nobag_backward_codegen_rowwise_adagrad_unweighted_"
            "pt2_wrapper("
            "    Tensor grad_output, "
            "    Tensor(a!) host_weights, "
            "    Tensor(b!) dev_weights, "
            "    Tensor(c!) uvm_weights, "
            "    Tensor(d!) lxu_cache_weights, "
            "    Tensor weights_placements, "
            "    Tensor weights_offsets, "
            "    SymInt D, "
            "    Tensor hash_size_cumsum, "
            "    int total_hash_size_bits, "
            "    Tensor indices, "
            "    Tensor offsets, "
            "    Tensor lxu_cache_locations, "
            "    int BT_block_size, "
            "    int max_segment_length_per_warp, "
            "    bool stochastic_rounding, "
            "    int info_B_num_bits, "
            "    int info_B_mask_int64, "
            "    bool use_uniq_cache_locations, "
            "    bool use_homogeneous_placements, "
            "    Tensor(g!) momentum1_host, "
            "    Tensor(h!) momentum1_dev, "
            "    Tensor(i!) momentum1_uvm, "
            "    Tensor momentum1_placements, "
            "    Tensor momentum1_offsets, "
            "    Tensor learning_rate_tensor, "
            "    float eps = 0, "
            "    float weight_decay = 0.0, "
            "    int weight_decay_mode = 0, "
            "    float max_norm = 0.0"
            ") -> Tensor");
    }

    if (!schemaExists("fbgemm::invert_permute")) {
        m.def("invert_permute(Tensor permute) -> Tensor");
    }

    if (!schemaExists("fbgemm::jagged_index_select_2d_forward")) {
        m.def(
            "jagged_index_select_2d_forward("
            "    Tensor values, "
            "    Tensor indices, "
            "    Tensor input_offsets, "
            "    Tensor output_offsets, "
            "    int num_dense_output_rows"
            ") -> Tensor");
    }

    if (!schemaExists("fbgemm::permute_1D_sparse_data")) {
        m.def(
            "permute_1D_sparse_data("
            "    Tensor permute, "
            "    Tensor lengths, "
            "    Tensor values, "
            "    Tensor? weights=None, "
            "    SymInt? permuted_lengths_sum=None"
            ") -> (Tensor, Tensor, Tensor?)");
    }

    if (!schemaExists("fbgemm::asynchronous_complete_cumsum")) {
        m.def("asynchronous_complete_cumsum(Tensor t_in) -> Tensor");
    }

    if (!schemaExists("fbgemm::permute_2D_sparse_data")) {
        m.def(
            "permute_2D_sparse_data("
            "    Tensor permute, "
            "    Tensor lengths, "
            "    Tensor values, "
            "    Tensor? weights=None, "
            "    SymInt? permuted_lengths_sum=None"
            ") -> (Tensor, Tensor, Tensor?)");
    }

    if (!schemaExists(
            "fbgemm::block_bucketize_sparse_features")) {
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
            ") -> (Tensor, Tensor, Tensor?, Tensor?, Tensor?)");
    }

    if (!schemaExists(
            "fbgemm::block_bucketize_sparse_features_inference")) {
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
            ") -> (Tensor, Tensor, Tensor?, Tensor?, Tensor?, Tensor?)");
    }

    if (!schemaExists("fbgemm::populate_bucketized_permute")) {
        m.def(
            "populate_bucketized_permute("
            "    Tensor lengths, "
            "    Tensor bucketized_lengths, "
            "    Tensor bucket_mapping"
            ") -> Tensor");
    }

    if (!schemaExists("fbgemm::reorder_batched_ad_lengths")) {
        m.def(
            "reorder_batched_ad_lengths("
            "    Tensor cat_ad_lengths, "
            "    Tensor batch_offsets, "
            "    int num_ads_in_batch, "
            "    bool broadcast_lengths=False, "
            "    int max_batch_size=0"
            ") -> Tensor");
    }

    if (!schemaExists("fbgemm::reorder_batched_ad_indices")) {
        m.def(
            "reorder_batched_ad_indices("
            "    Tensor cat_ad_offsets, "
            "    Tensor cat_ad_indices, "
            "    Tensor reordered_cat_ad_offsets, "
            "    Tensor batch_offsets, "
            "    int num_ads_in_batch, "
            "    bool broadcast_indices=False, "
            "    int num_indices_after_broadcast=-1"
            ") -> Tensor");
    }

    // ===== Jagged Tensor Operators =====

    if (!schemaExists("fbgemm::dense_to_jagged")) {
        m.def(
            "dense_to_jagged("
            "    Tensor dense, "
            "    Tensor[] x_offsets, "
            "    SymInt? total_L=None"
            ") -> (Tensor, Tensor[])");
    }

    if (!schemaExists("fbgemm::dense_to_jagged_forward")) {
        m.def(
            "dense_to_jagged_forward("
            "    Tensor dense, "
            "    Tensor[] x_offsets, "
            "    SymInt? total_L=None"
            ") -> Tensor");
    }

    if (!schemaExists("fbgemm::jagged_to_padded_dense")) {
        m.def(
            "jagged_to_padded_dense("
            "    Tensor values, "
            "    Tensor[] offsets, "
            "    SymInt[] max_lengths, "
            "    float padding_value=0.0"
            ") -> Tensor");
    }

    if (!schemaExists("fbgemm::jagged_to_padded_dense_forward")) {
        m.def(
            "jagged_to_padded_dense_forward("
            "    Tensor values, "
            "    Tensor[] offsets, "
            "    SymInt[] max_lengths, "
            "    float padding_value=0.0"
            ") -> Tensor");
    }

    if (!schemaExists("fbgemm::jagged_to_padded_dense_backward")) {
        m.def(
            "jagged_to_padded_dense_backward("
            "    Tensor grad_output, "
            "    Tensor[] offsets, "
            "    SymInt total_L"
            ") -> Tensor");
    }

    if (!schemaExists("fbgemm::jagged_dense_elementwise_add_jagged_output")) {
        m.def(
            "jagged_dense_elementwise_add_jagged_output("
            "    Tensor x_values, "
            "    Tensor[] x_offsets, "
            "    Tensor y"
            ") -> (Tensor, Tensor[])");
    }

    if (!schemaExists("fbgemm::jagged_2d_to_dense")) {
        m.def(
            "jagged_2d_to_dense("
            "    Tensor values, "
            "    Tensor offsets, "
            "    SymInt max_sequence_length"
            ") -> Tensor");
    }
}
