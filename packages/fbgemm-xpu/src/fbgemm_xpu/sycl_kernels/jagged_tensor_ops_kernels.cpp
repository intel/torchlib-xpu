/*
 * Copyright (c) Meta Platforms, Inc. and affiliates. All rights reserved.
 * Copyright (c) 2026 Intel Corporation. All Rights Reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */

#include <climits>
#include <cstdint>
#include <cstring>
#include <tuple>
#include <vector>

#include <ATen/ATen.h>
#include <ATen/Dispatch.h>
#include <ATen/native/StridedRandomAccessor.h>

#include "jagged_tensor_ops_kernels.h"

namespace syclext = sycl::ext::oneapi;
namespace syclexp = sycl::ext::oneapi::experimental;

using at::native::DefaultPtrTraits;
using at::native::RestrictPtrTraits;

namespace fbgemm_xpu {

// ============================================================================
// Shared helpers
// ============================================================================

constexpr size_t kStackArrayMaxDims = 5;

template <typename T>
struct StackArray {
  T vals[kStackArrayMaxDims];
  size_t ndim;
};

template <typename T, typename V>
inline auto CeilDivUp(T a, V b) {
  return (a + b - 1) / b;
}

template <typename T, typename V>
inline auto round_down(T a, V b) {
  return a / b * b;
}

template <typename T>
class SimpleAddFunctor3 {
 public:
  T operator()(T x, T y, T /*unused*/) {
    return x + y;
  }
};

template <typename T>
class SimpleRetSecondFunctor3 {
 public:
  T operator()(T /*unused*/, T y, T /*unused*/) {
    return y;
  }
};

template <typename T>
class SimpleRetFirstFunctor2 {
 public:
  T operator()(T x, T /*unused*/) {
    return x;
  }
};

// ----------------------------------------------------------------------------
// Vectorized helpers for the half-precision fast path of the add kernel.
// ----------------------------------------------------------------------------
struct VecType128 {
  typedef sycl::float4 TType; // Transaction Type
  typedef struct __attribute__((aligned(16))) {
    sycl::half a, b, c, d, w, x, y, z;
  } half8;

  union Data {
    half8 val;
    TType mask;
    Data() {
      mask = sycl::float4(0.0f, 0.0f, 0.0f, 0.0f);
    }
  } data;
};

struct VecType64 {
  typedef sycl::vec<float, 2> TType; // Transaction Type
  typedef struct __attribute__((aligned(8))) {
    sycl::half a, b, c, d;
  } half4;

  union Data {
    half4 val;
    TType mask;
    Data() {
      mask = sycl::vec<float, 2>(0.0f, 0.0f);
    }
  } data;
};

struct VecType32 {
  typedef float TType; // Transaction Type

  union Data {
    sycl::vec<sycl::half, 2> val;
    TType mask;
    Data() {
      mask = 0.0f;
    }
  } data;
};

template <typename F>
void f128(
    VecType128& v_out,
    const VecType128& x,
    const VecType128& y0,
    const VecType128& y1,
    F f) {
  v_out.data.val.a = f(x.data.val.a, y0.data.val.a, y1.data.val.a);
  v_out.data.val.b = f(x.data.val.b, y0.data.val.b, y1.data.val.b);
  v_out.data.val.c = f(x.data.val.c, y0.data.val.c, y1.data.val.c);
  v_out.data.val.d = f(x.data.val.d, y0.data.val.d, y1.data.val.d);
  v_out.data.val.w = f(x.data.val.w, y0.data.val.w, y1.data.val.w);
  v_out.data.val.x = f(x.data.val.x, y0.data.val.x, y1.data.val.x);
  v_out.data.val.y = f(x.data.val.y, y0.data.val.y, y1.data.val.y);
  v_out.data.val.z = f(x.data.val.z, y0.data.val.z, y1.data.val.z);
}

template <typename F>
void f64(
    VecType64& v_out,
    const VecType64& x,
    const VecType64& y0,
    const VecType64& y1,
    F f) {
  v_out.data.val.a = f(x.data.val.a, y0.data.val.a, y1.data.val.a);
  v_out.data.val.b = f(x.data.val.b, y0.data.val.b, y1.data.val.b);
  v_out.data.val.c = f(x.data.val.c, y0.data.val.c, y1.data.val.c);
  v_out.data.val.d = f(x.data.val.d, y0.data.val.d, y1.data.val.d);
}

template <typename F>
void f32(
    VecType32& v_out,
    const VecType32& x,
    const VecType32& y0,
    const VecType32& y1,
    F f) {
  v_out.data.val = sycl::vec<sycl::half, 2>(
      f(x.data.val.x(), y0.data.val.x(), y1.data.val.x()),
      f(x.data.val.y(), y0.data.val.y(), y1.data.val.y()));
}

template <typename F>
void fh(
    sycl::half& v_out,
    const sycl::half& x,
    const sycl::half& y0,
    const sycl::half& y1,
    F f) {
  v_out = f(x, y0, y1);
}

#define JAGGED_TENSOR_DISPATCH_DIMS()                                         \
  AT_DISPATCH_INDEX_TYPES(x_offsets[0].scalar_type(), "jagged_indices", [=] { \
    switch (num_jagged_dim) {                                                 \
      case 1:                                                                 \
        INVOKE_KERNEL_WITH_DIM(1);                                            \
        break;                                                                \
      case 2:                                                                 \
        INVOKE_KERNEL_WITH_DIM(2);                                            \
        break;                                                                \
      case 3:                                                                 \
        INVOKE_KERNEL_WITH_DIM(3);                                            \
        break;                                                                \
      case 4:                                                                 \
        INVOKE_KERNEL_WITH_DIM(4);                                            \
        break;                                                                \
      case 5:                                                                 \
        INVOKE_KERNEL_WITH_DIM(5);                                            \
        break;                                                                \
      default:                                                                \
        TORCH_CHECK(                                                          \
            false, "unsupported number of jagged dim ", num_jagged_dim);      \
    }                                                                         \
  });

// ============================================================================
// dense_to_jagged kernels
// ============================================================================

template <typename index_t, typename scalar_t, typename F>
SYCL_EXT_ONEAPI_FUNCTION_PROPERTY((syclexp::nd_range_kernel<2>))
void jagged_dense_elementwise_jagged_output_kernel_(
    const int num_jagged_dim,
    at::GenericPackedTensorAccessor<scalar_t, 2, DefaultPtrTraits, int32_t>
        x_values,
    StackArray<index_t*> x_offsets,
    StackArray<int64_t> x_offsets_sizes,
    at::GenericPackedTensorAccessor<scalar_t, 3, DefaultPtrTraits, int32_t>
        y_0,
    at::GenericPackedTensorAccessor<scalar_t, 3, DefaultPtrTraits, int32_t>
        y_1,
    at::GenericPackedTensorAccessor<scalar_t, 2, DefaultPtrTraits, int32_t>
        output_values,
    StackArray<int64_t> jagged_dims,
    F f) {
  auto output_values_acc = output_values;
  const int outer_dense_size = y_0.size(0);
  const int inner_dense_size = y_0.size(2);
  const int nnz = x_values.size(0);

  auto item = syclext::this_work_item::get_nd_item<2>();
  const int offset_begin =
      item.get_group(0) * item.get_local_range(1) + item.get_local_id(1);
  const int offset_stride = item.get_global_range(0) * item.get_local_range(1);
  for (int offset = offset_begin; offset < nnz; offset += offset_stride) {
    int offset_temp = offset;
    int jidx = 0;
    bool truncated = false;
    int dim_prod = 1;
#pragma unroll
    for (int d = num_jagged_dim - 1; d >= 0; --d) {
      // Binary search the first that is bigger than offset
      int count = x_offsets_sizes.vals[d] - 1;
      int first = 1;
      while (count > 0) {
        int idx = first;
        int step = count / 2;
        idx += step;
        if (x_offsets.vals[d][idx] <= offset_temp) {
          first = ++idx;
          count -= step + 1;
        } else {
          count = step;
        }
      }

      --first;
      int coord = offset_temp - x_offsets.vals[d][first];
      if (coord >= jagged_dims.vals[d]) {
        truncated = true;
        break;
      }
      jidx += coord * dim_prod;
      dim_prod *= jagged_dims.vals[d];
      offset_temp = first;
    }

    if (offset_temp >= outer_dense_size) {
      // This can happen when values have more elements than the last element of
      // offset
      truncated = true;
    }
    if (!truncated) {
      const int oidx = offset_temp;
      int iidx;
      for (iidx = item.get_local_id(0); iidx * 2 + 1 < inner_dense_size;
           iidx += item.get_local_range(0)) {
        output_values_acc[offset][2 * iidx] =
            f(x_values[offset][2 * iidx],
              y_0[oidx][jidx][2 * iidx],
              y_1[oidx][jidx][2 * iidx]);
        output_values_acc[offset][2 * iidx + 1] =
            f(x_values[offset][2 * iidx + 1],
              y_0[oidx][jidx][2 * iidx + 1],
              y_1[oidx][jidx][2 * iidx + 1]);
      }
      if (iidx * 2 + 1 == inner_dense_size) {
        output_values_acc[offset][2 * iidx] =
            f(x_values[offset][2 * iidx],
              y_0[oidx][jidx][2 * iidx],
              y_1[oidx][jidx][2 * iidx]);
      }
    } else {
      int iidx;
      for (iidx = item.get_local_id(0); iidx * 2 + 1 < inner_dense_size;
           iidx += item.get_local_range(0)) {
        output_values_acc[offset][2 * iidx] =
            f(x_values[offset][2 * iidx], 0, 0);
        output_values_acc[offset][2 * iidx + 1] =
            f(x_values[offset][2 * iidx + 1], 0, 0);
      }
      if (iidx * 2 + 1 == inner_dense_size) {
        output_values_acc[offset][2 * iidx] =
            f(x_values[offset][2 * iidx], 0, 0);
      }
    }
  }
}

template <int NUM_JAGGED_DIM, typename index_t>
bool walk_down_tensor_storage_tree_(
    int& offset,
    const int flattened_jagged_idx,
    const StackArray<int64_t>& jagged_dims,
    const StackArray<index_t*>& x_offsets) {
  // compute coorindates
  int jagged_coords[NUM_JAGGED_DIM];
  int j_temp = flattened_jagged_idx;
#pragma unroll
  for (int d = NUM_JAGGED_DIM - 1; d >= 0; --d) {
    const int jagged_size = jagged_dims.vals[d];
    jagged_coords[d] = j_temp % jagged_size;
    j_temp /= jagged_size;
  }

  // walk down the tree
  bool is_zero = false;
#pragma unroll
  for (int d = 0; d < NUM_JAGGED_DIM; ++d) {
    const int begin = x_offsets.vals[d][offset];
    const int end = x_offsets.vals[d][offset + 1];
    if (jagged_coords[d] >= end - begin) {
      is_zero = true;
      break;
    }
    offset = begin + jagged_coords[d];
  }
  return is_zero;
}

inline std::tuple<int64_t, int64_t, int64_t, StackArray<int64_t>>
check_shape_and_partition_(
    const at::Tensor& values,
    const std::vector<at::Tensor>& offsets,
    const at::Tensor& dense_tensor) {
  const int32_t outer_dense_size = dense_tensor.size(0);
  TORCH_CHECK(
      outer_dense_size == offsets[0].numel() - 1,
      "outer_dense_size, ",
      outer_dense_size,
      " != offsets[0].numel() - 1, ",
      offsets[0].numel() - 1);
  const int32_t inner_dense_size = dense_tensor.size(-1);
  TORCH_CHECK(
      inner_dense_size == values.size(-1),
      "inner_dense_size, ",
      inner_dense_size,
      " != values.size(-1), ",
      values.size(-1));
  const int32_t jagged_folded_size =
      dense_tensor.numel() / (outer_dense_size * inner_dense_size);

  const int32_t sub_group_size = syclMaxSubGroupSize();
  const int64_t wg_size_0 = inner_dense_size >= sub_group_size / 2
      ? sub_group_size
      : inner_dense_size;
  const int64_t wg_size_1 = syclDeviceMaxWorkGroupSize() / sub_group_size;
  const int64_t wg_num =
      CeilDivUp(outer_dense_size * jagged_folded_size, (int32_t)wg_size_1);

  StackArray<int64_t> jagged_dims_tensor;
  const int32_t num_jagged_dim = dense_tensor.dim() - 2;
  TORCH_CHECK(num_jagged_dim <= kStackArrayMaxDims);
  jagged_dims_tensor.ndim = num_jagged_dim;
  std::memcpy(
      &(jagged_dims_tensor.vals[0]),
      dense_tensor.sizes().data() + 1,
      num_jagged_dim * sizeof(int64_t));
  return {wg_size_0, wg_size_1, wg_num, jagged_dims_tensor};
}

template <typename scalar_t, typename F>
void jagged_dense_elementwise_jagged_output_(
    const at::Tensor& x_values,
    const std::vector<at::Tensor>& x_offsets,
    const at::Tensor& y,
    const at::Tensor& output_values,
    F f) {
  // Canonicalize y to 3D, collapsing jagged dimensions.
  const int num_jagged_dim = y.dim() - 2;
  const at::Tensor y_reshaped = y.view({y.size(0), -1, y.size(-1)});

  int64_t wg_0, wg_1, wg_num;
  StackArray<int64_t> jagged_dims_tensor;
  std::tie(wg_0, wg_1, wg_num, jagged_dims_tensor) =
      check_shape_and_partition_(x_values, x_offsets, y);
  wg_num = CeilDivUp(x_values.size(0), wg_1);
  std::vector<at::Tensor> x_offsets_contig;
  x_offsets_contig.resize(num_jagged_dim);
  StackArray<int64_t*> x_offset_ptrs;
  x_offset_ptrs.ndim = num_jagged_dim;
  StackArray<int64_t> x_offset_sizes;
  x_offset_sizes.ndim = num_jagged_dim;
  for (int d = 0; d < num_jagged_dim; ++d) {
    x_offsets_contig[d] = x_offsets[d].contiguous();
    x_offset_ptrs.vals[d] = x_offsets_contig[d].template data_ptr<int64_t>();
    x_offset_sizes.vals[d] = x_offsets[d].numel();
  }
  sycl_kernel_submit<
      jagged_dense_elementwise_jagged_output_kernel_<int64_t, scalar_t, F>>(
      sycl::range<2>(wg_0 * wg_num, wg_1),
      sycl::range<2>(wg_0, wg_1),
      at::xpu::getCurrentSYCLQueue(),
      0,
      num_jagged_dim,
      x_values.packed_accessor32<scalar_t, 2, DefaultPtrTraits>(),
      x_offset_ptrs,
      x_offset_sizes,
      y_reshaped.packed_accessor32<scalar_t, 3, DefaultPtrTraits>(),
      y_reshaped.packed_accessor32<scalar_t, 3, DefaultPtrTraits>(),
      output_values.packed_accessor32<scalar_t, 2, DefaultPtrTraits>(),
      jagged_dims_tensor,
      f);
}

void dense_to_jagged_forward_xpu_kernel(
    const at::Tensor& x_values,
    const std::vector<at::Tensor>& x_offsets,
    const at::Tensor& y,
    const at::Tensor& output_values) {
  AT_DISPATCH_ALL_TYPES_AND2(
      at::ScalarType::BFloat16,
      at::ScalarType::Half,
      x_values.scalar_type(),
      "dense_to_jagged_forward_xpu_kernel",
      [&]() {
        jagged_dense_elementwise_jagged_output_<scalar_t>(
            x_values,
            x_offsets,
            y,
            output_values,
            SimpleRetSecondFunctor3<scalar_t>());
      });
}

// ============================================================================
// jagged_to_padded_dense kernels
// ============================================================================

template <int NUM_JAGGED_DIM, typename index_t, typename scalar_t, typename F>
SYCL_EXT_ONEAPI_FUNCTION_PROPERTY((syclexp::nd_range_kernel<2>))
void jagged_dense_elementwise_dense_output_kernel_(
    at::GenericPackedTensorAccessor<scalar_t, 2, DefaultPtrTraits, int32_t>
        x_values,
    StackArray<index_t*> x_offsets,
    at::GenericPackedTensorAccessor<scalar_t, 3, DefaultPtrTraits, int32_t>
        y,
    at::GenericPackedTensorAccessor<scalar_t, 3, DefaultPtrTraits, int32_t>
        output,
    StackArray<int64_t> jagged_dims,
    const scalar_t padding_value,
    F f) {
  auto output_acc = output;
  const int outer_dense_size = y.size(0);
  const int jagged_folded_size = y.size(1);
  const int inner_dense_size = y.size(2);

  auto item = syclext::this_work_item::get_nd_item<2>();
  const int outer_begin =
      item.get_group(0) * item.get_local_range(1) + item.get_local_id(1);
  const int outer_stride = item.get_global_range(0) * item.get_local_range(1);
  for (int outer = outer_begin; outer < outer_dense_size * jagged_folded_size;
       outer += outer_stride) {
    const int oidx = outer / jagged_folded_size;
    const int jidx = outer % jagged_folded_size;

    int offset = oidx;
    const bool is_zero = walk_down_tensor_storage_tree_<NUM_JAGGED_DIM>(
        offset, jidx, jagged_dims, x_offsets);

    if (is_zero) {
      int iidx;
      for (iidx = item.get_local_id(0); iidx * 2 + 1 < inner_dense_size;
           iidx += item.get_local_range(0)) {
        output_acc[oidx][jidx][2 * iidx] =
            f(padding_value, y[oidx][jidx][2 * iidx]);
        output_acc[oidx][jidx][2 * iidx + 1] =
            f(padding_value, y[oidx][jidx][2 * iidx + 1]);
      }
      if (iidx * 2 + 1 == inner_dense_size) {
        output_acc[oidx][jidx][2 * iidx] =
            f(padding_value, y[oidx][jidx][2 * iidx]);
      }
    } else {
      int iidx;
      for (iidx = item.get_local_id(0); iidx * 2 + 1 < inner_dense_size;
           iidx += item.get_local_range(0)) {
        output_acc[oidx][jidx][2 * iidx] =
            f(x_values[offset][2 * iidx], y[oidx][jidx][2 * iidx]);
        output_acc[oidx][jidx][2 * iidx + 1] =
            f(x_values[offset][2 * iidx + 1], y[oidx][jidx][2 * iidx + 1]);
      }
      if (iidx * 2 + 1 == inner_dense_size) {
        output_acc[oidx][jidx][2 * iidx] =
            f(x_values[offset][2 * iidx], y[oidx][jidx][2 * iidx]);
      }
    }
  }
}

template <int NUM_JAGGED_DIM, typename index_t, typename scalar_t>
void jagged_dense_elementwise_dense_output_launch_(
    const at::GenericPackedTensorAccessor<
        scalar_t,
        2,
        DefaultPtrTraits,
        int32_t> x_values,
    StackArray<index_t*> x_offsets,
    const at::GenericPackedTensorAccessor<
        scalar_t,
        3,
        DefaultPtrTraits,
        int32_t> y,
    at::GenericPackedTensorAccessor<scalar_t, 3, DefaultPtrTraits, int32_t>
        output,
    StackArray<int64_t> jagged_dims,
    const scalar_t padding_value,
    int64_t wg_0,
    int64_t wg_1,
    int64_t wg_num) {
  sycl_kernel_submit<jagged_dense_elementwise_dense_output_kernel_<
      NUM_JAGGED_DIM,
      index_t,
      scalar_t,
      SimpleRetFirstFunctor2<scalar_t>>>(
      sycl::range<2>(wg_0 * wg_num, wg_1),
      sycl::range<2>(wg_0, wg_1),
      at::xpu::getCurrentSYCLQueue(),
      0,
      x_values,
      x_offsets,
      y,
      output,
      jagged_dims,
      padding_value,
      SimpleRetFirstFunctor2<scalar_t>());
}

template <typename scalar_t>
void jagged_dense_elementwise_dense_output_(
    const at::Tensor& x_values,
    const std::vector<at::Tensor>& x_offsets,
    const at::Tensor& y,
    const at::Tensor& output,
    const scalar_t padding_value) {
  int64_t wg_0, wg_1, wg_num;
  StackArray<int64_t> jagged_dims_tensor;
  std::tie(wg_0, wg_1, wg_num, jagged_dims_tensor) =
      check_shape_and_partition_(x_values, x_offsets, y);

  // Canonicalize y and output to 3D, collapsing jagged dimensions.
  const int num_jagged_dim = y.dim() - 2;
  const at::Tensor y_reshaped = y.view({y.size(0), -1, y.size(-1)});
  at::Tensor output_reshaped = output.view(y_reshaped.sizes());

#define INVOKE_KERNEL_WITH_DIM(NUM_JAGGED_DIM)                              \
  {                                                                         \
    std::vector<at::Tensor> x_offsets_contig;                               \
    x_offsets_contig.resize(num_jagged_dim);                                \
    StackArray<index_t*> x_offset_ptrs;                                     \
    x_offset_ptrs.ndim = num_jagged_dim;                                    \
    for (int d = 0; d < num_jagged_dim; ++d) {                              \
      x_offsets_contig[d] = x_offsets[d].contiguous();                      \
      x_offset_ptrs.vals[d] =                                               \
          x_offsets_contig[d].template data_ptr<index_t>();                 \
    }                                                                       \
    jagged_dense_elementwise_dense_output_launch_<NUM_JAGGED_DIM, index_t>( \
        x_values.packed_accessor32<scalar_t, 2, DefaultPtrTraits>(),        \
        x_offset_ptrs,                                                      \
        y_reshaped.packed_accessor32<scalar_t, 3, DefaultPtrTraits>(),      \
        output_reshaped.packed_accessor32<scalar_t, 3, DefaultPtrTraits>(), \
        jagged_dims_tensor,                                                 \
        padding_value,                                                      \
        wg_0,                                                               \
        wg_1,                                                               \
        wg_num);                                                            \
  }

  JAGGED_TENSOR_DISPATCH_DIMS();
#undef INVOKE_KERNEL_WITH_DIM
}

void jagged_to_padded_dense_forward_xpu_kernel(
    const at::Tensor& x_values,
    const std::vector<at::Tensor>& x_offsets,
    const at::Tensor& y,
    const at::Tensor& output,
    const double padding_value) {
  AT_DISPATCH_ALL_TYPES_AND2(
      at::ScalarType::BFloat16,
      at::ScalarType::Half,
      x_values.scalar_type(),
      "jagged_to_padded_dense_forward_xpu_kernel",
      [&] {
        jagged_dense_elementwise_dense_output_<scalar_t>(
            x_values,
            x_offsets,
            y, // not used
            output,
            static_cast<scalar_t>(padding_value));
      });
}

// ============================================================================
// jagged_dense_elementwise_add kernels
// ============================================================================

// Check to see if the inputs to the op are amenable to the fast path
inline bool jagged_dense_dense_elementwise_jagged_output_matches_opt(
    const int& num_jagged_dim,
    const at::Tensor& x_values,
    const std::vector<at::Tensor>& x_offsets,
    const at::Tensor& y_0_reshaped,
    const at::Tensor& y_1_reshaped,
    const at::Tensor& output_values) {
  bool matches = true;
  matches &= (num_jagged_dim == 1);

  // Unit stride embedding dim
  matches &= (x_values.stride(-1) == 1);
  matches &= (output_values.stride(-1) == 1);
  matches &= (y_0_reshaped.stride(-1) == 1);
  matches &= (y_1_reshaped.stride(-1) == 1);

  // Each row is aligned to 128-bit
  matches &= ((x_values.stride(-2) & 0x7) == 0);
  matches &= ((output_values.stride(-2) & 0x7) == 0);
  matches &= ((y_0_reshaped.stride(-2) & 0x7) == 0);
  matches &= ((y_1_reshaped.stride(-2) & 0x7) == 0);

  // Base addresses aligned to 128-bit
  matches &= ((reinterpret_cast<uint64_t>(x_values.data_ptr()) & 0xF) == 0);
  matches &=
      ((reinterpret_cast<uint64_t>(output_values.data_ptr()) % 0xF) == 0);
  matches &= ((reinterpret_cast<uint64_t>(y_0_reshaped.data_ptr()) % 0xF) == 0);
  matches &= ((reinterpret_cast<uint64_t>(y_1_reshaped.data_ptr()) % 0xF) == 0);

  // Rows and col fit into int32_t
  matches &= (y_0_reshaped.size(0) < INT_MAX);
  matches &= (y_0_reshaped.size(1) < INT_MAX);

  // maximum shared local memory size
  int max_shared_bytes = syclLocalMemSize();
  // Use all shared memory, no L1 cache consideration
  int max_shared_kb = max_shared_bytes >> 10;
  int used_shared_kb = round_down(max_shared_kb, 16);
  TORCH_CHECK(used_shared_kb > 0);
  int used_shared_bytes = used_shared_kb << 10;
  AT_DISPATCH_INDEX_TYPES(
      x_offsets[0].scalar_type(), "check_shared_memory", [&] {
        auto B = y_0_reshaped.size(0);
        if ((B + 1) * sizeof(index_t) >= used_shared_bytes) {
          matches = false;
        }
      });
  return matches;
}

template <typename index_t>
SYCL_EXT_ONEAPI_FUNCTION_PROPERTY((syclexp::nd_range_kernel<1>))
void jagged_dense_dense_elementwise_jagged_output_opt_search_kernel_(
    const at::GenericPackedTensorAccessor<index_t, 1, RestrictPtrTraits, int32_t>
        offsets,
    at::GenericPackedTensorAccessor<int, 1, RestrictPtrTraits, int32_t> rows,
    at::GenericPackedTensorAccessor<int, 1, RestrictPtrTraits, int32_t> cols,
    int nnz,
    int B) {
  index_t* offsets_sh =
      reinterpret_cast<index_t*>(syclexp::get_work_group_scratch_memory());
  auto item = syclext::this_work_item::get_nd_item<1>();

  for (auto i = item.get_local_id(0); i < B + 1; i += item.get_local_range(0)) {
    offsets_sh[i] = offsets[i];
  }
  group_barrier(item.get_group());
  auto row = item.get_local_id(0) + item.get_group(0) * item.get_local_range(0);
  if (row >= nnz)
    return;
  int first = 1;
  int count = B - 1;
  while (count > 0) {
    int idx = first;
    int step = count / 2;
    idx += step;
    if (offsets_sh[idx] <= row) {
      first = ++idx;
      count -= step + 1;
    } else {
      count = step;
    }
  }
  --first;

  int dense_row = first;
  int offset = offsets_sh[dense_row];
  int dense_col = row - offset;
  rows[row] = dense_row;
  cols[row] = dense_col;
}

template <typename index_t, typename F>
SYCL_EXT_ONEAPI_FUNCTION_PROPERTY((syclexp::nd_range_kernel<2>))
void jagged_dense_dense_elementwise_jagged_output_opt_gather_kernel_(
    at::GenericPackedTensorAccessor<c10::Half, 2, RestrictPtrTraits, int32_t>
        values,
    const at::GenericPackedTensorAccessor<
        c10::Half,
        2,
        RestrictPtrTraits,
        int32_t> x_values,
    const at::GenericPackedTensorAccessor<
        c10::Half,
        3,
        RestrictPtrTraits,
        int32_t> y0,
    const at::GenericPackedTensorAccessor<
        c10::Half,
        3,
        RestrictPtrTraits,
        int32_t> y1,
    const at::GenericPackedTensorAccessor<int, 1, RestrictPtrTraits, int32_t>
        rows,
    const at::GenericPackedTensorAccessor<int, 1, RestrictPtrTraits, int32_t>
        cols,
    const int nnz,
    const int E,
    F f) {
  auto item = syclext::this_work_item::get_nd_item<2>();
  auto values_row =
      item.get_local_id(1) + item.get_group(1) * item.get_local_range(1);
  if (values_row >= nnz)
    return;
  for (int real_row = values_row; real_row < nnz;
       real_row += item.get_local_range(1) * item.get_group_range(1)) {
    int dense_row = rows[real_row];
    int dense_col = cols[real_row];
    sycl::half* values_ptr =
        reinterpret_cast<sycl::half*>(&values[real_row][0]);
    const sycl::half* x_ptr =
        reinterpret_cast<const sycl::half*>(&x_values[real_row][0]);
    const sycl::half* y0_ptr =
        reinterpret_cast<const sycl::half*>(&y0[dense_row][dense_col][0]);
    const sycl::half* y1_ptr =
        reinterpret_cast<const sycl::half*>(&y1[dense_row][dense_col][0]);
    if ((dense_col < y0.size(1)) && (dense_row < y0.size(0)) &&
        (dense_col < y1.size(1)) && (dense_row < y1.size(0)) &&
        (dense_col >= 0) && (dense_row >= 0)) {
      for (auto tid = item.get_local_id(1); tid < E / 8;
           tid += item.get_local_range(1)) {
        VecType128 v_x, v_out, v_y0, v_y1;
        v_x.data.mask =
            (reinterpret_cast<const VecType128::TType*>(x_ptr))[tid];
        v_y0.data.mask =
            (reinterpret_cast<const VecType128::TType*>(y0_ptr))[tid];
        v_y1.data.mask =
            (reinterpret_cast<const VecType128::TType*>(y1_ptr))[tid];
        f128(v_out, v_x, v_y0, v_y1, f);
        (reinterpret_cast<VecType128::TType*>(values_ptr))[tid] =
            v_out.data.mask;
      }
      for (auto tid = item.get_local_id(1) + (E / 8) * 8; tid < E / 4;
           tid += item.get_local_range(1)) {
        VecType64 v_x, v_out, v_y0, v_y1;
        v_x.data.mask = (reinterpret_cast<const VecType64::TType*>(x_ptr))[tid];
        v_y0.data.mask =
            (reinterpret_cast<const VecType64::TType*>(y0_ptr))[tid];
        v_y1.data.mask =
            (reinterpret_cast<const VecType64::TType*>(y1_ptr))[tid];
        f64(v_out, v_x, v_y0, v_y1, f);
        (reinterpret_cast<VecType64::TType*>(values_ptr))[tid] =
            v_out.data.mask;
      }
      for (auto tid = item.get_local_id(1) + (E / 4) * 4; tid < E / 2;
           tid += item.get_local_range(1)) {
        VecType32 v_x, v_out, v_y0, v_y1;
        v_x.data.mask = (reinterpret_cast<const VecType32::TType*>(x_ptr))[tid];
        v_y0.data.mask =
            (reinterpret_cast<const VecType32::TType*>(y0_ptr))[tid];
        v_y1.data.mask =
            (reinterpret_cast<const VecType32::TType*>(y1_ptr))[tid];
        f32(v_out, v_x, v_y0, v_y1, f);
        (reinterpret_cast<VecType32::TType*>(values_ptr))[tid] =
            v_out.data.mask;
      }
      for (auto tid = item.get_local_id(1) + (E / 2) * 2; tid < E;
           tid += item.get_local_range(1)) {
        sycl::half v_x, v_out, v_y0, v_y1;
        v_x = static_cast<sycl::half>(x_ptr[tid]);
        v_y0 = static_cast<sycl::half>(y0_ptr[tid]);
        v_y1 = static_cast<sycl::half>(y1_ptr[tid]);
        fh(v_out, v_x, v_y0, v_y1, f);
        values_ptr[tid] = v_out;
      }
    } else {
      for (auto tid = item.get_local_id(1); tid < E / 8;
           tid += item.get_local_range(1)) {
        VecType128 v_x, v_out, v_y0, v_y1;
        v_x.data.mask =
            (reinterpret_cast<const VecType128::TType*>(x_ptr))[tid];
        f128(v_out, v_x, v_y0, v_y1, f);
        (reinterpret_cast<VecType128::TType*>(values_ptr))[tid] =
            v_out.data.mask;
      }
      for (auto tid = item.get_local_id(1) + (E / 8) * 8; tid < E / 4;
           tid += item.get_local_range(1)) {
        VecType64 v_x, v_out, v_y0, v_y1;
        v_x.data.mask = (reinterpret_cast<const VecType64::TType*>(x_ptr))[tid];
        f64(v_out, v_x, v_y0, v_y1, f);
        (reinterpret_cast<VecType64::TType*>(values_ptr))[tid] =
            v_out.data.mask;
      }
      for (auto tid = item.get_local_id(1) + (E / 4) * 4; tid < E / 2;
           tid += item.get_local_range(1)) {
        VecType32 v_x, v_out, v_y0, v_y1;
        v_x.data.mask = (reinterpret_cast<const VecType32::TType*>(x_ptr))[tid];
        f32(v_out, v_x, v_y0, v_y1, f);
        (reinterpret_cast<VecType32::TType*>(values_ptr))[tid] =
            v_out.data.mask;
      }
      for (auto tid = item.get_local_id(1) + (E / 2) * 2; tid < E;
           tid += item.get_local_range(1)) {
        sycl::half v_x, v_out, v_y0, v_y1;
        v_x = static_cast<sycl::half>(x_ptr[tid]);
        fh(v_out, v_x, v_y0, v_y1, f);
        values_ptr[tid] = v_out;
      }
    }
  }
}

template <typename index_t, typename scalar_t, typename F>
SYCL_EXT_ONEAPI_FUNCTION_PROPERTY((syclexp::nd_range_kernel<2>))
void jagged_dense_dense_elementwise_jagged_output_kernel_(
    const int num_jagged_dim,
    const at::GenericPackedTensorAccessor<
        scalar_t,
        2,
        RestrictPtrTraits,
        int32_t> x_values,
    StackArray<index_t*> x_offsets,
    StackArray<int64_t> x_offsets_sizes,
    const at::GenericPackedTensorAccessor<
        scalar_t,
        3,
        RestrictPtrTraits,
        int32_t> y_0,
    const at::GenericPackedTensorAccessor<
        scalar_t,
        3,
        RestrictPtrTraits,
        int32_t> y_1,
    at::GenericPackedTensorAccessor<scalar_t, 2, RestrictPtrTraits, int32_t>
        output_values,
    StackArray<int64_t> jagged_dims,
    F f) {
  const int outer_dense_size = y_0.size(0);
  const int inner_dense_size = y_0.size(2);
  const int nnz = x_values.size(0);

  auto item = syclext::this_work_item::get_nd_item<2>();

  const auto offset_begin =
      item.get_group(0) * item.get_local_range(1) + item.get_local_id(1);
  const auto offset_stride = item.get_group_range(0) * item.get_local_range(1);
  for (int offset = offset_begin; offset < nnz; offset += offset_stride) {
    int offset_temp = offset;
    int jidx = 0;
    bool truncated = false;
    int dim_prod = 1;
#pragma unroll
    for (int d = num_jagged_dim - 1; d >= 0; --d) {
      // Binary search the first that is bigger than offset
      int count = x_offsets_sizes.vals[d] - 1;
      int first = 1;
      while (count > 0) {
        int idx = first;
        int step = count / 2;
        idx += step;
        if (x_offsets.vals[d][idx] <= offset_temp) {
          first = ++idx;
          count -= step + 1;
        } else {
          count = step;
        }
      }

      --first;
      int coord = offset_temp - x_offsets.vals[d][first];
      if (coord >= jagged_dims.vals[d]) {
        truncated = true;
        break;
      }
      jidx += coord * dim_prod;
      dim_prod *= jagged_dims.vals[d];
      offset_temp = first;
    }

    if (offset_temp >= outer_dense_size) {
      // This can happen when values have more elements than the last element of
      // offset
      truncated = true;
    }
    if (!truncated) {
      const int oidx = offset_temp;
      int iidx;
      for (iidx = item.get_local_id(0); iidx * 2 + 1 < inner_dense_size;
           iidx += item.get_local_range(0)) {
        output_values[offset][2 * iidx] =
            f(x_values[offset][2 * iidx],
              y_0[oidx][jidx][2 * iidx],
              y_1[oidx][jidx][2 * iidx]);
        output_values[offset][2 * iidx + 1] =
            f(x_values[offset][2 * iidx + 1],
              y_0[oidx][jidx][2 * iidx + 1],
              y_1[oidx][jidx][2 * iidx + 1]);
      }
      if (iidx * 2 + 1 == inner_dense_size) {
        output_values[offset][2 * iidx] =
            f(x_values[offset][2 * iidx],
              y_0[oidx][jidx][2 * iidx],
              y_1[oidx][jidx][2 * iidx]);
      }
    } else {
      int iidx;
      for (iidx = item.get_local_id(0); iidx * 2 + 1 < inner_dense_size;
           iidx += item.get_local_range(0)) {
        output_values[offset][2 * iidx] = f(x_values[offset][2 * iidx], 0, 0);
        output_values[offset][2 * iidx + 1] =
            f(x_values[offset][2 * iidx + 1], 0, 0);
      }
      if (iidx * 2 + 1 == inner_dense_size) {
        output_values[offset][2 * iidx] = f(x_values[offset][2 * iidx], 0, 0);
      }
    }
  }
}

template <typename scalar_t, typename F>
void jagged_dense_elementwise_jagged_output_opt_(
    const at::Tensor& x_values,
    const std::vector<at::Tensor>& x_offsets,
    const at::Tensor& y,
    const at::Tensor& output_values,
    F f) {
  // Canonicalize y to 3D, collapsing jagged dimensions.
  const int num_jagged_dim = y.dim() - 2;
  const at::Tensor y_reshaped = y.view({y.size(0), -1, y.size(-1)});
  if (jagged_dense_dense_elementwise_jagged_output_matches_opt(
          num_jagged_dim,
          x_values,
          x_offsets,
          y_reshaped,
          y_reshaped,
          output_values)) {
    auto nnz = output_values.size(0);
    auto B = y_reshaped.size(0);
    auto E = y_reshaped.size(2);
    at::Tensor t_rows_after_bs = at::empty(
        {nnz},
        at::TensorOptions().dtype(at::kInt).device(
            at::kXPU, c10::xpu::current_device()));
    at::Tensor t_cols_after_bs = at::empty(
        {nnz},
        at::TensorOptions().dtype(at::kInt).device(
            at::kXPU, c10::xpu::current_device()));

    // Binary search
    size_t dynamic_smem_size = (B + 1) * sizeof(int64_t);
    auto max_shared_bytes = syclLocalMemSize();
    int max_shared_kb = max_shared_bytes >> 10;
    int used_shared_kb = round_down(max_shared_kb, 16);
    TORCH_CHECK(used_shared_kb > 0);
    int used_shared_bytes = used_shared_kb << 10;
    TORCH_CHECK(dynamic_smem_size <= used_shared_bytes);

    int max_wg_size = syclDeviceMaxWorkGroupSize();
    int wg_size = max_wg_size < 1024 ? max_wg_size : 1024;

    int nbr_of_wg = CeilDivUp(nnz, wg_size);
    sycl_kernel_submit<
        jagged_dense_dense_elementwise_jagged_output_opt_search_kernel_<
            int64_t>>(
        sycl::range<1>(nbr_of_wg * wg_size),
        sycl::range<1>(wg_size),
        at::xpu::getCurrentSYCLQueue(),
        dynamic_smem_size,
        x_offsets[0].packed_accessor32<int64_t, 1, RestrictPtrTraits>(),
        t_rows_after_bs.packed_accessor32<int, 1, RestrictPtrTraits>(),
        t_cols_after_bs.packed_accessor32<int, 1, RestrictPtrTraits>(),
        nnz,
        B);

    // Gather kernel
    int dim_0_1 = 16;
    int nbr_of_wg_g = CeilDivUp(nnz, dim_0_1);
    if (nbr_of_wg_g > 65535) {
      nbr_of_wg_g = round_down(65535, dim_0_1);
    }

    sycl_kernel_submit<
        jagged_dense_dense_elementwise_jagged_output_opt_gather_kernel_<
            int64_t,
            F>>(
        sycl::range<2>(dim_0_1, dim_0_1 * nbr_of_wg_g),
        sycl::range<2>(dim_0_1, dim_0_1),
        at::xpu::getCurrentSYCLQueue(),
        0,
        output_values.packed_accessor32<c10::Half, 2, RestrictPtrTraits>(),
        x_values.packed_accessor32<c10::Half, 2, RestrictPtrTraits>(),
        y_reshaped.packed_accessor32<c10::Half, 3, RestrictPtrTraits>(),
        y_reshaped.packed_accessor32<c10::Half, 3, RestrictPtrTraits>(),
        t_rows_after_bs.packed_accessor32<int, 1, RestrictPtrTraits>(),
        t_cols_after_bs.packed_accessor32<int, 1, RestrictPtrTraits>(),
        nnz,
        E,
        f);
  } else {
    int64_t wg_0, wg_1, wg_num;
    StackArray<int64_t> jagged_dims_tensor;
    std::tie(wg_0, wg_1, wg_num, jagged_dims_tensor) =
        check_shape_and_partition_(x_values, x_offsets, y);
    wg_num = CeilDivUp(x_values.size(0), wg_1);
    std::vector<at::Tensor> x_offsets_contig;
    x_offsets_contig.resize(num_jagged_dim);
    StackArray<int64_t*> x_offset_ptrs;
    x_offset_ptrs.ndim = num_jagged_dim;
    StackArray<int64_t> x_offset_sizes;
    x_offset_sizes.ndim = num_jagged_dim;
    for (int d = 0; d < num_jagged_dim; ++d) {
      x_offsets_contig[d] = x_offsets[d].contiguous();
      x_offset_ptrs.vals[d] = x_offsets_contig[d].template data_ptr<int64_t>();
      x_offset_sizes.vals[d] = x_offsets[d].numel();
    }
    sycl_kernel_submit<jagged_dense_dense_elementwise_jagged_output_kernel_<
        int64_t,
        scalar_t,
        F>>(
        sycl::range<2>(wg_0 * wg_num, wg_1),
        sycl::range<2>(wg_0, wg_1),
        at::xpu::getCurrentSYCLQueue(),
        0,
        num_jagged_dim,
        x_values.packed_accessor32<scalar_t, 2, RestrictPtrTraits>(),
        x_offset_ptrs,
        x_offset_sizes,
        y_reshaped.packed_accessor32<scalar_t, 3, RestrictPtrTraits>(),
        y_reshaped.packed_accessor32<scalar_t, 3, RestrictPtrTraits>(),
        output_values.packed_accessor32<scalar_t, 2, RestrictPtrTraits>(),
        jagged_dims_tensor,
        f);
  }
}

void jagged_dense_elementwise_add_jagged_output_fwd_xpu_kn(
    const at::Tensor& x_values,
    const std::vector<at::Tensor>& offsets,
    const at::Tensor& dense,
    const at::Tensor& output_values) {
  if (x_values.scalar_type() == at::ScalarType::Half) {
    jagged_dense_elementwise_jagged_output_opt_<c10::Half>(
        x_values,
        offsets,
        dense,
        output_values,
        SimpleAddFunctor3<sycl::half>());
  } else if (x_values.scalar_type() == at::ScalarType::Float) {
    jagged_dense_elementwise_jagged_output_<float>(
        x_values, offsets, dense, output_values, SimpleAddFunctor3<float>());
  } else {
    jagged_dense_elementwise_jagged_output_<c10::BFloat16>(
        x_values,
        offsets,
        dense,
        output_values,
        SimpleAddFunctor3<c10::BFloat16>());
  }
}

} // namespace fbgemm_xpu
