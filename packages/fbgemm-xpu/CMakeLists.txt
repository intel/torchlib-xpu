cmake_minimum_required(VERSION 3.18)
project(${SKBUILD_PROJECT_NAME})

add_subdirectory(src/fbgemm_xpu)
add_subdirectory(src/codegen)
