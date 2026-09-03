/*
 * Copyright (c) Meta Platforms, Inc. and affiliates. All rights reserved.
 * Copyright (c) 2026 Intel Corporation. All Rights Reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */

#pragma once

#include <torch/library.h>

#define PRIVATE_CASE_TYPE_CACHE(enum_type, type, ...) \
  case enum_type: {                                   \
    using cache_t = type;                             \
    return __VA_ARGS__();                             \
  }


#define PRIVATE_CASE_TYPE_EMB(enum_type1, enum_type2, type1, NAME, ...)    \
  case enum_type1: {                                                       \
    using emb_t = type1;                                                   \
    switch (enum_type2) {                                                  \
      PRIVATE_CASE_TYPE_CACHE(at::ScalarType::Float, float, __VA_ARGS__)   \
      PRIVATE_CASE_TYPE_CACHE(at::ScalarType::Half, at::Half, __VA_ARGS__) \
      default:                                                             \
        AT_ERROR(                                                          \
            #NAME,                                                         \
            " not implemented for cache_t '",                              \
            toString(enum_type2),                                          \
            "'");                                                          \
    }                                                                      \
  }

#define _DISPATCH_EMB_CACHE_TYPES(emb_enum_type, cache_enum_type, NAME, ...)  \
  at::ScalarType _emb_t = emb_enum_type;                                      \
  at::ScalarType _cache_t = cache_enum_type;                                  \
  switch (_emb_t) {                                                           \
    PRIVATE_CASE_TYPE_EMB(                                                    \
        at::ScalarType::Float, _cache_t, float, NAME, __VA_ARGS__)            \
    PRIVATE_CASE_TYPE_EMB(                                                    \
        at::ScalarType::Half, _cache_t, at::Half, NAME, __VA_ARGS__)          \
    default:                                                                  \
      AT_ERROR(#NAME, " not implemented for emb_t '", toString(_emb_t), "'"); \
  }


#define PRIVATE_CASE_TYPE_OUTPUT(                            \
    output_enum_type1,                                       \
    emb_enum_type1,                                          \
    cache_enum_type1,                                        \
    output_type1,                                            \
    NAME,                                                    \
    ...)                                                     \
  case output_enum_type1: {                                  \
    using output_t = output_type1;                           \
    _DISPATCH_EMB_CACHE_TYPES(                               \
        emb_enum_type1, cache_enum_type1, NAME, __VA_ARGS__) \
  }

#define DISPATCH_EMB_CACHE_OUTPUT_TYPES(                           \
    EMB_TYPE, CACHE_TYPE, OUTPUT_TYPE, NAME, ...)                  \
  [&] {                                                            \
    const at::ScalarType _output_t = OUTPUT_TYPE; /*            */ \
    const auto& emb_type = EMB_TYPE;                               \
    const auto& cache_type = CACHE_TYPE;                           \
    switch (_output_t) {                                           \
      PRIVATE_CASE_TYPE_OUTPUT(                                    \
          at::ScalarType::Half,                                    \
          emb_type,                                                \
          cache_type,                                              \
          at::Half,                                                \
          NAME,                                                    \
          __VA_ARGS__)                                             \
      PRIVATE_CASE_TYPE_OUTPUT(                                    \
          at::ScalarType::Float,                                   \
          emb_type,                                                \
          cache_type,                                              \
          float,                                                   \
          NAME,                                                    \
          __VA_ARGS__)                                             \
      PRIVATE_CASE_TYPE_OUTPUT(                                    \
          at::ScalarType::BFloat16,                                \
          emb_type,                                                \
          cache_type,                                              \
          at::BFloat16,                                            \
          NAME,                                                    \
          __VA_ARGS__)                                             \
      default:                                                     \
        AT_ERROR(                                                  \
            #NAME,                                                 \
            " not implemented for output_t '",                     \
            toString(_output_t),                                   \
            "'");                                                  \
    }                                                              \
  }()

#define PRIVATE_CASE_TYPE_CACHE_EMB(                                       \
    grad_enum_type, _cache_t, _emb_t, grad_cxx_type, NAME, ...)            \
  case grad_enum_type: {                                                   \
    using grad_t = grad_cxx_type;                                          \
    switch (_emb_t) {                                                      \
      PRIVATE_CASE_TYPE_EMB(                                               \
          at::ScalarType::Float, _cache_t, float, NAME, __VA_ARGS__)       \
      PRIVATE_CASE_TYPE_EMB(                                               \
          at::ScalarType::Half, _cache_t, at::Half, NAME, __VA_ARGS__)     \
      default:                                                             \
        AT_ERROR(                                                          \
            #NAME, " not implemented for emb_t '", toString(_emb_t), "'"); \
    }                                                                      \
  }

#define DISPATCH_EMB_GRAD_CACHE_TYPES(                                         \
    EMB_TYPE, GRAD_TYPE, CACHE_TYPE, NAME, ...)                                \
  [&] {                                                                        \
    const auto& emb_type = EMB_TYPE;                                           \
    const auto& grad_type = GRAD_TYPE;                                         \
    const auto& cache_type = CACHE_TYPE;                                       \
    at::ScalarType _emb_t = emb_type;                                          \
    at::ScalarType _grad_t = grad_type;                                        \
    at::ScalarType _cache_t = cache_type;                                      \
    switch (_grad_t) {                                                         \
      PRIVATE_CASE_TYPE_CACHE_EMB(                                             \
          at::ScalarType::Float, _cache_t, _emb_t, float, NAME, __VA_ARGS__)   \
      PRIVATE_CASE_TYPE_CACHE_EMB(                                             \
          at::ScalarType::Half, _cache_t, _emb_t, at::Half, NAME, __VA_ARGS__) \
      PRIVATE_CASE_TYPE_CACHE_EMB(                                             \
          at::ScalarType::BFloat16,                                            \
          _cache_t,                                                            \
          _emb_t,                                                              \
          at::BFloat16,                                                        \
          NAME,                                                                \
          __VA_ARGS__)                                                         \
      default:                                                                 \
        AT_ERROR(                                                              \
            #NAME, " not implemented for grad_t '", toString(_grad_t), "'");   \
    }                                                                          \
  }()

// ============================================================================
// Type Dispatch Macros for Sparse Operations
// ============================================================================

#define FBGEMM_DISPATCH_FLOATING_TYPES_CASE(...)       \
  AT_DISPATCH_CASE(at::ScalarType::Float, __VA_ARGS__) \
  AT_DISPATCH_CASE(at::ScalarType::Half, __VA_ARGS__)  \
  AT_DISPATCH_CASE(at::ScalarType::BFloat16, __VA_ARGS__)

#define FBGEMM_DISPATCH_INTEGRAL_TYPES_CASE(...)     \
  AT_DISPATCH_CASE(at::ScalarType::Int, __VA_ARGS__) \
  AT_DISPATCH_CASE(at::ScalarType::Long, __VA_ARGS__)

#define FBGEMM_DISPATCH_ALL_TYPES(TYPE, NAME, ...)     \
  AT_DISPATCH_SWITCH(                                  \
      TYPE,                                            \
      NAME,                                            \
      FBGEMM_DISPATCH_FLOATING_TYPES_CASE(__VA_ARGS__) \
          FBGEMM_DISPATCH_INTEGRAL_TYPES_CASE(__VA_ARGS__))

// Add new dispatch families only together with a generated-host user.
