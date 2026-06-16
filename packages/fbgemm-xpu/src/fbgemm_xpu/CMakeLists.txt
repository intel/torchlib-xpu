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
# SYCL backend: required for fbgemm-xpu. All kernels are SYCL-based, so the
# Intel oneAPI compiler (icpx) is mandatory.
# --------------------------------------------------------------------------
if(NOT CMAKE_CXX_COMPILER MATCHES "icpx")
    message(FATAL_ERROR "fbgemm-xpu requires Intel oneAPI compiler (icpx) with SYCL support")
endif()
message(STATUS "Intel compiler detected: SYCL kernels enabled")

# --------------------------------------------------------------------------
# XPU architecture: currently only PVC (Ponte Vecchio) is supported.
# --------------------------------------------------------------------------
if(NOT "$ENV{TORCH_XPU_ARCH_LIST}" STREQUAL "")
    set(TORCH_XPU_ARCH_LIST "$ENV{TORCH_XPU_ARCH_LIST}")
else()
    set(TORCH_XPU_ARCH_LIST "pvc")
endif()
message(STATUS "Building for XPU architectures: ${TORCH_XPU_ARCH_LIST}")

set(SYCL_TARGETS -fsycl-targets=spir64_gen,spir64)
set(SYCL_DEVICE_LIST -Xs "-device ${TORCH_XPU_ARCH_LIST} -options -cl-poison-unsupported-fp64-kernels")

# --------------------------------------------------------------------------
# Sources: both host and SYCL kernel sources must be present for fbgemm-xpu
# to function. SYCL kernels (.sycl) and companion .cpp files are collected
# from sycl_kernels/ and fbgemm_utils/ directories.
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

set(all_sources ${host_sources} ${sycl_sources_cxx})

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
)

target_compile_options(_C PRIVATE -fsycl ${SYCL_TARGETS})
target_link_options(_C PRIVATE -fsycl ${SYCL_TARGETS} ${SYCL_DEVICE_LIST})

install(TARGETS _C DESTINATION fbgemm_xpu)
