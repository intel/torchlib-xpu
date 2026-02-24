# Copyright (c) 2025 Dmitry Rogozhkin.

cmake_minimum_required(VERSION 3.18)
project(TorchCodec)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(PYBIND11_FINDPYTHON ON)
find_package(pybind11 REQUIRED)
find_package(Torch REQUIRED)
find_package(TorchCodec REQUIRED)
find_package(Python3 ${PYTHON_VERSION} EXACT COMPONENTS Development)

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
    set(libname "libtorchcodec_xpu${torchcodec_variant}")
    set(sources
        XpuDeviceInterface.cpp)

    add_library(${libname} SHARED ${sources})
    # Avoid adding the "lib" prefix which we already add explicitly.
    set_target_properties(${libname} PROPERTIES PREFIX "")
    set_target_properties(${libname} PROPERTIES CXX_STANDARD 17)
    target_include_directories(${libname}
        PRIVATE
        "${Python3_INCLUDE_DIRS}"
    )
    target_link_libraries(${libname}
      ${TORCH_LIBRARIES}
      torchcodec::core${torchcodec_variant}
      torchcodec::ffmpeg${torchcodec_variant}
      PkgConfig::L0
    )

    if(WIN32)
        # On Windows, the built artifacts are put in Release/Debug
        # subdirectories by default. We want to avoid that, otherwise our
        # install() step would not know where to find those.
        set_target_properties(${libname} PROPERTIES
            RUNTIME_OUTPUT_DIRECTORY_DEBUG ${CMAKE_CURRENT_BINARY_DIR}
            RUNTIME_OUTPUT_DIRECTORY_RELEASE ${CMAKE_CURRENT_BINARY_DIR}
            LIBRARY_OUTPUT_DIRECTORY_DEBUG ${CMAKE_CURRENT_BINARY_DIR}
            LIBRARY_OUTPUT_DIRECTORY_RELEASE ${CMAKE_CURRENT_BINARY_DIR}
            ARCHIVE_OUTPUT_DIRECTORY_DEBUG ${CMAKE_CURRENT_BINARY_DIR}
            ARCHIVE_OUTPUT_DIRECTORY_RELEASE ${CMAKE_CURRENT_BINARY_DIR}
        )
    endif()

    # The install step is invoked within CMakeBuild.build_library() in
    # setup.py and just copies the built files from the temp
    # cmake/setuptools build folder into the CMAKE_INSTALL_PREFIX folder. We
    # still need to manually pass "DESTINATION ..." for cmake to copy those
    # files in CMAKE_INSTALL_PREFIX instead of CMAKE_INSTALL_PREFIX/lib.
    install(
        TARGETS ${libname}
        LIBRARY DESTINATION ${CMAKE_INSTALL_PREFIX}
        RUNTIME DESTINATION ${CMAKE_INSTALL_PREFIX}  # For Windows DLLs
    )
endfunction()

foreach(variant IN LISTS TORCHCODEC_VARIANTS)
    make_torchcodec_xpu_libraries(${variant})
endforeach()
