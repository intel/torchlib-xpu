# --------------------------------------------------------------------------
# Sources: keep build inputs explicit so new files are reviewed when added.
# Add new host and SYCL kernel .cpp files to host_sources as operators are
# integrated.
# --------------------------------------------------------------------------
set(host_sources
    ${CMAKE_CURRENT_SOURCE_DIR}/ops_registry.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/sparse_async_cumsum.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/split_embeddings_utils_xpu.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/sycl_kernels/block_bucketize_sparse_features.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/sycl_kernels/dense_to_jagged_forward.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/sycl_kernels/embedding_bounds_check_kernel.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/sycl_kernels/expand_into_jagged_permute_kernel.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/sycl_kernels/invert_permute_kernel.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/sycl_kernels/jagged_dense_elementwise_add_jagged_output.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/sycl_kernels/jagged_index_select_2d.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/sycl_kernels/jagged_to_padded_dense_backward.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/sycl_kernels/jagged_to_padded_dense_forward.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/sycl_kernels/permute_1d_sparse_data.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/sycl_kernels/permute_2d_sparse_data.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/sycl_kernels/reorder_batched_ad.cpp
)

set(all_sources ${host_sources})

# --------------------------------------------------------------------------
# Extension module: fbgemm._C
#
# SYCL kernels are required. Skip target creation if no sources have been
# added yet, this lets the scaffolding configure cleanly before any code
# is written.
# --------------------------------------------------------------------------
if(NOT all_sources)
    message(WARNING "fbgemm._C: no sources defined yet; skipping extension target.")
    return()
endif()

Python3_add_library(_C MODULE WITH_SOABI ${all_sources})
set_target_properties(_C PROPERTIES PREFIX "")
set_target_properties(_C PROPERTIES CXX_STANDARD 20)

# The package root is needed because host sources include
# "fbgemm_utils/torch_library.h". sycl_kernels and fbgemm_utils are added so
# future kernel .cpp files can resolve same-dir headers
# ("feature_gates.h", "utils.h", ...).
target_include_directories(_C PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/sycl_kernels
    ${CMAKE_CURRENT_SOURCE_DIR}/fbgemm_utils
    ${Python3_INCLUDE_DIRS}
)

target_link_libraries(_C PRIVATE ${TORCH_LIBRARIES})

target_compile_options(_C PRIVATE
    -fdiagnostics-color=always
)

target_compile_options(_C PRIVATE -fsycl ${SYCL_TARGETS})
target_link_options(_C PRIVATE -fsycl ${SYCL_TARGETS} ${SYCL_DEVICE_LIST})

install(TARGETS _C DESTINATION fbgemm_xpu)
