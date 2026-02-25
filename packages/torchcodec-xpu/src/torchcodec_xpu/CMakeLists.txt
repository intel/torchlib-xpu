# Copyright (c) 2025 Dmitry Rogozhkin.

cmake_minimum_required(VERSION 3.18)
project(${SKBUILD_PROJECT_NAME})
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(PYBIND11_FINDPYTHON ON)
find_package(pybind11 REQUIRED)
find_package(Torch REQUIRED)
find_package(TorchCodec REQUIRED)
find_package(Python3 EXACT COMPONENTS Development)

if(DEFINED TORCHCODEC_XPU_DISABLE_COMPILE_WARNING_AS_ERROR AND TORCHCODEC_XPU_DISABLE_COMPILE_WARNING_AS_ERROR)
    set(TORCHCODEC_XPU_WERROR_OPTION "")
else()
    if (WIN32)
        # TODO set warnings as errors on Windows as well.
        # set(TORCHCODEC_XPU_WERROR_OPTION "/WX")
    else()
        set(TORCHCODEC_XPU_WERROR_OPTION "-Werror")
    endif()
endif()

if (WIN32)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /W4 ${TORCHCODEC_XPU_WERROR_OPTION} ${TORCH_CXX_FLAGS}")
else()
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -pedantic ${TORCHCODEC_XPU_WERROR_OPTION} ${TORCH_CXX_FLAGS}")
endif()

find_package(PkgConfig REQUIRED)
pkg_check_modules(L0 REQUIRED IMPORTED_TARGET level-zero)
pkg_check_modules(LIBVA REQUIRED IMPORTED_TARGET libva)

function(make_torchcodec_xpu_libraries torchcodec_variant)
    set(libname "xpu_ops${torchcodec_variant}")
    set(sources
        XpuDeviceInterface.cpp)

    python_add_library(${libname} MODULE WITH_SOABI ${sources})
    # Avoid adding the "lib" prefix which we already add explicitly.
    set_target_properties(${libname} PROPERTIES PREFIX "")
    set_target_properties(${libname} PROPERTIES CXX_STANDARD 17)
    target_include_directories(${libname}
        PRIVATE
        "${Python3_INCLUDE_DIRS}"
    )
    target_link_libraries(${libname} PRIVATE
      ${TORCH_LIBRARIES}
      torchcodec::core${torchcodec_variant}
      torchcodec::ffmpeg${torchcodec_variant}
      PkgConfig::L0
    )

    install(
        TARGETS ${libname}
	DESTINATION ${SKBUILD_PROJECT_NAME}
    )
endfunction()

foreach(variant IN LISTS TORCHCODEC_VARIANTS)
    make_torchcodec_xpu_libraries(${variant})
endforeach()
