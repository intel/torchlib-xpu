# Copyright (c) 2026 Intel Corporation. All Rights Reserved.
# SPDX-License-Identifier: BSD-3-Clause

cmake_minimum_required(VERSION 3.18)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# --------------------------------------------------------------------------
# Python / Torch / pybind11 — mirror the setup in src/fbgemm_xpu to ensure
# the same interpreter is selected when building both sub-directories.
# --------------------------------------------------------------------------
set(PYBIND11_FINDPYTHON ON)
set(Python3_FIND_VIRTUALENV FIRST)
if(DEFINED ENV{VIRTUAL_ENV})
    set(Python3_ROOT_DIR "$ENV{VIRTUAL_ENV}")
    set(Python3_EXECUTABLE "$ENV{VIRTUAL_ENV}/bin/python")
endif()

find_package(Python3 REQUIRED COMPONENTS Interpreter Development.Module)
find_package(pybind11 REQUIRED)
find_package(Torch REQUIRED)

# --------------------------------------------------------------------------
# SYCL backend: Intel oneAPI compiler (icpx) is mandatory.
# --------------------------------------------------------------------------
if(NOT CMAKE_CXX_COMPILER MATCHES "icpx")
    message(FATAL_ERROR "fbgemm-xpu requires Intel oneAPI compiler (icpx) with SYCL support")
endif()
message(STATUS "[codegen] Intel compiler detected: SYCL kernels enabled")

# --------------------------------------------------------------------------
# XPU architecture — defaults to PVC (Ponte Vecchio).
# --------------------------------------------------------------------------
if(NOT "$ENV{TORCH_XPU_ARCH_LIST}" STREQUAL "")
    set(TORCH_XPU_ARCH_LIST "$ENV{TORCH_XPU_ARCH_LIST}")
else()
    set(TORCH_XPU_ARCH_LIST "pvc")
endif()
message(STATUS "[codegen] Building for XPU architectures: ${TORCH_XPU_ARCH_LIST}")

set(SYCL_TARGETS -fsycl-targets=spir64_gen,spir64)
set(SYCL_DEVICE_LIST -Xs "-device ${TORCH_XPU_ARCH_LIST} -options -cl-poison-unsupported-fp64-kernels")

# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------
set(GENSCRIPT_DIR      ${CMAKE_CURRENT_SOURCE_DIR}/genscript)
set(TRAINING_DIR       ${CMAKE_CURRENT_SOURCE_DIR}/training)
set(FBGEMM_XPU_SRC_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../fbgemm_xpu)

# All generated sources land under a per-build-tree directory so that
# parallel build trees never clobber one another.
set(CODEGEN_OUTPUT_DIR ${CMAKE_CURRENT_BINARY_DIR}/generated)

# --------------------------------------------------------------------------
# Forward-split code generation
#
# generate_forward_split.py writes files relative to --install_dir, placing
# .cpp/.h files inside the sycl_kernels/ sub-directory:
#
#   <install_dir>/sycl_kernels/gen_embedding_forward_*
# --------------------------------------------------------------------------
set(FORWARD_GEN_DIR ${CODEGEN_OUTPUT_DIR}/sycl_kernels)

set(FORWARD_GENERATED_SOURCES
    ${FORWARD_GEN_DIR}/gen_embedding_forward_dense_unweighted_nobag_host.cpp
    ${FORWARD_GEN_DIR}/gen_embedding_forward_split_unweighted_nobag_host.cpp
    ${FORWARD_GEN_DIR}/gen_embedding_forward_split_unweighted_nobag_pt2_wrapper.cpp
)

set(FORWARD_GENERATED_HEADERS
    ${FORWARD_GEN_DIR}/gen_embedding_forward_dense_unweighted_nobag_kernel.h
    ${FORWARD_GEN_DIR}/gen_embedding_forward_dense_unweighted_nobag_kernel_small.h
    ${FORWARD_GEN_DIR}/gen_embedding_forward_split_unweighted_nobag_kernel.h
    ${FORWARD_GEN_DIR}/gen_embedding_forward_split_unweighted_nobag_kernel_small.h
)

add_custom_command(
    OUTPUT
        ${FORWARD_GENERATED_SOURCES}
        ${FORWARD_GENERATED_HEADERS}
    COMMAND
        ${CMAKE_COMMAND} -E make_directory ${FORWARD_GEN_DIR}
    COMMAND
        ${Python3_EXECUTABLE}
            ${GENSCRIPT_DIR}/generate_forward_split.py
            --opensource
            --install_dir ${CODEGEN_OUTPUT_DIR}
    DEPENDS
        ${GENSCRIPT_DIR}/common.py
        ${GENSCRIPT_DIR}/generate_forward_split.py
        ${GENSCRIPT_DIR}/jinja_environment.py
        ${GENSCRIPT_DIR}/scripts_argsparse.py
        ${GENSCRIPT_DIR}/torch_type_utils.py
        ${TRAINING_DIR}/forward/embedding_forward_nobag_unweighted_host_template.cpp
        ${TRAINING_DIR}/forward/embedding_forward_split_kernel_nobag_small_template.h
        ${TRAINING_DIR}/forward/embedding_forward_split_kernel_template.h
        ${TRAINING_DIR}/pt2/embedding_forward_nobag_unweighted_pt2_wrapper_template.cpp
    COMMENT "Generating forward split embedding SYCL sources"
    VERBATIM
)

# --------------------------------------------------------------------------
# Backward-split code generation
#
# generate_backward_split.py uses the same --install_dir interface as the
# forward generator and writes files under the sycl_kernels/ sub-directory.
# --------------------------------------------------------------------------
set(BACKWARD_GENERATED_SOURCES
    ${FORWARD_GEN_DIR}/gen_embedding_backward_dense_unweighted_nobag_host.cpp
    ${FORWARD_GEN_DIR}/gen_embedding_backward_rowwise_adagrad_split_unweighted_nobag_host.cpp
    ${FORWARD_GEN_DIR}/gen_embedding_backward_split_rowwise_adagrad_nobag_unweighted_pt2_xpu_wrapper.cpp
)

set(BACKWARD_GENERATED_HEADERS
    ${FORWARD_GEN_DIR}/gen_embedding_backward_dense_unweighted_nobag_kernels.h
    ${FORWARD_GEN_DIR}/gen_embedding_backward_rowwise_adagrad_split_unweighted_nobag_kernels.h
)

add_custom_command(
    OUTPUT
        ${BACKWARD_GENERATED_SOURCES}
        ${BACKWARD_GENERATED_HEADERS}
    COMMAND
        ${CMAKE_COMMAND} -E make_directory ${FORWARD_GEN_DIR}
    COMMAND
        ${Python3_EXECUTABLE}
            ${GENSCRIPT_DIR}/generate_backward_split.py
            --opensource
            --install_dir ${CODEGEN_OUTPUT_DIR}
    DEPENDS
        ${GENSCRIPT_DIR}/common.py
        ${GENSCRIPT_DIR}/generate_backward_split.py
        ${GENSCRIPT_DIR}/jinja_environment.py
        ${GENSCRIPT_DIR}/scripts_argsparse.py
        ${GENSCRIPT_DIR}/torch_type_utils.py
        ${TRAINING_DIR}/backward/embedding_backward_nobag_unweighted_host_template.cpp
        ${TRAINING_DIR}/backward/embedding_backward_split_kernel_templates.h
        ${TRAINING_DIR}/pt2/embedding_forward_nobag_unweighted_pt2_wrapper_template.cpp
    COMMENT "Generating backward split embedding SYCL sources"
    VERBATIM
)

# Convenience target that drives both generators — downstream targets depend
# on the individual outputs, but this target is useful for manual rebuilds.
add_custom_target(generate_lookup_kernels
    DEPENDS
        ${FORWARD_GENERATED_SOURCES}
        ${FORWARD_GENERATED_HEADERS}
        ${BACKWARD_GENERATED_SOURCES}
        ${BACKWARD_GENERATED_HEADERS}
)

# --------------------------------------------------------------------------
# Minimal Python module init for _C_training
#
# The training module registers all operators via TORCH_LIBRARY_IMPL static
# initialisers; no Python-callable functions are exposed.  We still need a
# valid PyInit function so the .so can be imported (or loaded) by the
# Python package __init__.py.
#
# file(GENERATE ...) creates the file at CMake-generate time so it is
# always present when the compiler is invoked.
# --------------------------------------------------------------------------
set(CODEGEN_REGISTRY_CPP ${CMAKE_CURRENT_BINARY_DIR}/codegen_registry.cpp)

file(GENERATE
    OUTPUT ${CODEGEN_REGISTRY_CPP}
    CONTENT
"// Auto-generated by src/codegen/CMakeLists.txt — do not edit.
#include <Python.h>

extern \"C\" {
PyObject* PyInit__C_training(void) {
    static struct PyModuleDef module_def = {
        PyModuleDef_HEAD_INIT,
        \"_C_training\",  /* module name */
        nullptr,          /* docstring */
        -1,               /* per-interpreter state size */
        nullptr,          /* method table */
    };
    return PyModule_Create(&module_def);
}
} // extern \"C\"
"
)

# --------------------------------------------------------------------------
# Sources
# --------------------------------------------------------------------------
set(training_sources
    # Auto-generated init stub
    ${CODEGEN_REGISTRY_CPP}
    # Static operator files (forward/backward host wrappers + autograd fns)
    ${TRAINING_DIR}/ops/fbgemm_dense_lookups_ops.cpp
    ${TRAINING_DIR}/ops/fbgemm_split_lookups_ops.cpp
    # Generated forward kernel host files
    ${FORWARD_GENERATED_SOURCES}
    # Generated backward kernel host files
    ${BACKWARD_GENERATED_SOURCES}
    # Utility sources from src/fbgemm_xpu needed by the training operators
    ${FBGEMM_XPU_SRC_DIR}/fbgemm_utils/feature_gates.cpp
    ${FBGEMM_XPU_SRC_DIR}/fbgemm_utils/utils.cpp
    # SYCL kernel implementations for backward utilities:
    #   - SplitEmbeddingBackwardFindLongSegments::operator()
    #   - transpose_embedding_input()
    # Declared in backward_utils.h, defined here (must be compiled with -fsycl).
    ${FBGEMM_XPU_SRC_DIR}/fbgemm_utils/backward_utils.cpp
    # Prefix-sum kernel used by backward_utils (asynchronous_complete_cumsum_xpu).
    ${FBGEMM_XPU_SRC_DIR}/sparse_async_cumsum.cpp
)

# --------------------------------------------------------------------------
# Extension module: fbgemm_xpu._C_training
# --------------------------------------------------------------------------
Python3_add_library(_C_training MODULE WITH_SOABI ${training_sources})
set_target_properties(_C_training PROPERTIES
    PREFIX      ""
    CXX_STANDARD 17
)

# Ensure generated files are produced before compilation starts.
add_dependencies(_C_training generate_lookup_kernels)

target_include_directories(_C_training PRIVATE
    # Headers from sibling fbgemm_xpu directory (fbgemm_utils/, sycl_kernels/)
    ${FBGEMM_XPU_SRC_DIR}
    ${FBGEMM_XPU_SRC_DIR}/sycl_kernels
    ${FBGEMM_XPU_SRC_DIR}/fbgemm_utils
    # Generated kernel headers
    ${FORWARD_GEN_DIR}
    ${Python3_INCLUDE_DIRS}
)

# sparse_async_cumsum.cpp is shared with the _C extension, which owns
# the fbgemm::asynchronous_complete_cumsum operator registration. Here we only
# need the asynchronous_complete_cumsum_xpu() C++ helper for backward_utils, so
# suppress the duplicate TORCH_LIBRARY_IMPL registration in this target.
target_compile_definitions(_C_training PRIVATE FBGEMM_XPU_TRAINING_BUILD)

target_link_libraries(_C_training PRIVATE ${TORCH_LIBRARIES})

target_compile_options(_C_training PRIVATE
    -fdiagnostics-color=always
    -fsycl
    ${SYCL_TARGETS}
)

target_link_options(_C_training PRIVATE
    -fsycl
    ${SYCL_TARGETS}
    ${SYCL_DEVICE_LIST}
)

install(TARGETS _C_training DESTINATION fbgemm_xpu)
