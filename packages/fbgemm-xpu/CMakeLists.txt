cmake_minimum_required(VERSION 3.18)
project(${SKBUILD_PROJECT_NAME})

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(PYBIND11_FINDPYTHON ON)
set(Python3_FIND_VIRTUALENV FIRST)
if(DEFINED ENV{VIRTUAL_ENV})
    set(Python3_ROOT_DIR "$ENV{VIRTUAL_ENV}")
    set(Python3_EXECUTABLE "$ENV{VIRTUAL_ENV}/bin/python")
endif()

find_package(Python3 REQUIRED COMPONENTS Interpreter Development.Module)
find_package(pybind11 REQUIRED)
find_package(Torch REQUIRED)

if(NOT CMAKE_CXX_COMPILER MATCHES "icpx")
    message(FATAL_ERROR "fbgemm-xpu requires Intel oneAPI compiler (icpx) with SYCL support")
endif()
message(STATUS "Intel compiler detected: SYCL kernels enabled")

if(NOT "$ENV{TORCH_XPU_ARCH_LIST}" STREQUAL "")
    set(TORCH_XPU_ARCH_LIST "$ENV{TORCH_XPU_ARCH_LIST}")
else()
    set(TORCH_XPU_ARCH_LIST "pvc")
endif()
message(STATUS "Building for XPU architectures: ${TORCH_XPU_ARCH_LIST}")

set(SYCL_TARGETS -fsycl-targets=spir64_gen,spir64)
set(SYCL_DEVICE_LIST -Xs "-device ${TORCH_XPU_ARCH_LIST} -options -cl-poison-unsupported-fp64-kernels")

add_subdirectory(src/fbgemm_xpu)
add_subdirectory(src/codegen)
