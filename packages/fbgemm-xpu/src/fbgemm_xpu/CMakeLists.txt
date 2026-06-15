cmake_minimum_required(VERSION 3.18)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(PYBIND11_FINDPYTHON ON)
# Prefer the active virtual environment Python when available.
set(Python3_FIND_VIRTUALENV FIRST)
if(DEFINED ENV{VIRTUAL_ENV})
    set(Python3_ROOT_DIR "$ENV{VIRTUAL_ENV}")
    set(Python3_EXECUTABLE "$ENV{VIRTUAL_ENV}/bin/python")
endif()

find_package(Python3 REQUIRED COMPONENTS Interpreter Development.Module)
find_package(pybind11 REQUIRED)
find_package(Torch REQUIRED)

# --------------------------------------------------------------------------
# SYCL backend detection: enabled only when CXX is the Intel oneAPI compiler
# (icpx). Falls back to a host-only CPU build otherwise. Mirrors the previous
# setup.py behavior driven by USE_SYCL / torch.xpu.is_available().
# --------------------------------------------------------------------------
if(CMAKE_CXX_COMPILER MATCHES "icpx")
    set(WITH_SYCL ON)
    message(STATUS "Intel compiler detected: SYCL kernels enabled")
else()
    set(WITH_SYCL OFF)
    message(STATUS "Non-Intel compiler in use: SYCL kernels disabled")
endif()

# --------------------------------------------------------------------------
# XPU architecture list. Honors TORCH_XPU_ARCH_LIST env var, otherwise queries
# torch.xpu.get_arch_list(), otherwise defaults to "pvc".
# --------------------------------------------------------------------------
if(NOT "$ENV{TORCH_XPU_ARCH_LIST}" STREQUAL "")
    set(TORCH_XPU_ARCH_LIST "$ENV{TORCH_XPU_ARCH_LIST}")
else()
    execute_process(
        COMMAND ${Python3_EXECUTABLE} -c "import torch; print(','.join(torch.xpu.get_arch_list()))"
        OUTPUT_VARIABLE TORCH_XPU_ARCH_LIST
        OUTPUT_STRIP_TRAILING_WHITESPACE
        RESULT_VARIABLE _arch_rc
    )
    if(NOT _arch_rc EQUAL 0)
        message(WARNING "Failed to retrieve torch.xpu.get_arch_list()")
    endif()
endif()
message(STATUS "Building for XPU architectures: ${TORCH_XPU_ARCH_LIST}")

set(SYCL_TARGETS -fsycl-targets=spir64_gen,spir64)
set(SYCL_DEVICE_LIST -Xs "-device ${TORCH_XPU_ARCH_LIST} -options -cl-poison-unsupported-fp64-kernels")

# --------------------------------------------------------------------------
# Sources
#
# host_sources lists host C++ files that live alongside this CMakeLists.
# SYCL kernels (.sycl) and companion .cpp files under sycl_kernels/ or
# fbgemm_utils/ are picked up automatically by the GLOB_RECURSE below.
# --------------------------------------------------------------------------
# Auto-include all host C++ sources in this directory.
file(GLOB host_sources CONFIGURE_DEPENDS
    *.cpp
)

file(GLOB_RECURSE sycl_sources CONFIGURE_DEPENDS
    sycl_kernels/*.sycl
    sycl_kernels/*.cpp
    fbgemm_utils/*.sycl
    fbgemm_utils/*.cpp
)

# CMake does not recognize the .sycl extension as a C++ source. Mirror every
# .sycl file into the build tree with a .cpp suffix so CMake schedules a
# proper compile step for it; the icpx driver then handles SYCL via -fsycl.
set(sycl_sources_cxx "")
foreach(_src IN LISTS sycl_sources)
    if(_src MATCHES "\\.sycl$")
        file(RELATIVE_PATH _rel ${CMAKE_CURRENT_SOURCE_DIR} ${_src})
        set(_mirrored ${CMAKE_CURRENT_BINARY_DIR}/${_rel}.cpp)
        configure_file(${_src} ${_mirrored} COPYONLY)
        list(APPEND sycl_sources_cxx ${_mirrored})
    else()
        list(APPEND sycl_sources_cxx ${_src})
    endif()
endforeach()

if(WITH_SYCL)
    set(all_sources ${host_sources} ${sycl_sources_cxx})
else()
    set(all_sources ${host_sources})
endif()

# --------------------------------------------------------------------------
# Extension module: fbgemm._C
#
# Skip target creation if no sources have been added yet — this lets the
# scaffolding configure cleanly before any code is written.
# --------------------------------------------------------------------------
if(NOT all_sources)
    message(WARNING "fbgemm._C: no sources defined yet; skipping extension target.")
    return()
endif()

Python3_add_library(_C MODULE WITH_SOABI ${all_sources})
set_target_properties(_C PROPERTIES PREFIX "")
set_target_properties(_C PROPERTIES CXX_STANDARD 17)

# The package root is needed because host sources include
# "fbgemm_utils/torch_library.h". sycl_kernels and fbgemm_utils are added so
# SYCL sources mirrored into the build tree can still resolve same-dir headers
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
    -Wno-c++11-narrowing
)

if(WITH_SYCL)
    target_compile_options(_C PRIVATE -fsycl ${SYCL_TARGETS})
    target_link_options(_C PRIVATE -fsycl ${SYCL_TARGETS} ${SYCL_DEVICE_LIST})
endif()

install(TARGETS _C DESTINATION fbgemm_xpu)
