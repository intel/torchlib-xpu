/*
 * Copyright (c) Meta Platforms, Inc. and affiliates. All rights reserved.
 * Copyright (c) 2026 Intel Corporation. All Rights Reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */

#include "permute_2d_sparse_data.h"

namespace syclext = sycl::ext::oneapi;
namespace syclexp = sycl::ext::oneapi::experimental;

// ============================================================================
// fbgemm_cumsum_kernel implementation
// This is from the original FbgemmKernels.sycl
// ============================================================================

namespace at {
namespace native {
namespace xpu {

void fbgemm_cumsum_kernel(
    const Tensor& result,
    const Tensor& self,
    int64_t dim) {
  AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND2(
      ScalarType::Half,
      ScalarType::BFloat16,
      self.scalar_type(),
      "fbgemm_cumsum_xpu",
      [&]() {
        scalar_t init = 0;
        scan<INCLUSIVE_TYPE, const scalar_t, scalar_t>(
            result, self, dim, init, std::plus<scalar_t>());
      });
}

} // namespace xpu
} // namespace native
} // namespace at

// ============================================================================
// fbgemm_xpu namespace implementation
// ============================================================================

namespace fbgemm_xpu {

// ============================================================================
// Asynchronous Exclusive Cumsum Helper
// ============================================================================

at::Tensor asynchronous_exclusive_cumsum(const at::Tensor& t_in) {
  TENSOR_ON_SYCL_XPU(t_in);
  SYCL_DEVICE_GUARD(t_in);

  if (t_in.numel() == 0) {
    return at::empty_like(t_in);
  }

  TORCH_CHECK(t_in.is_contiguous());
  TORCH_CHECK(t_in.dtype() == at::kInt || t_in.dtype() == at::kLong);
  // only handles up to INT_MAX elements.
  TORCH_CHECK(t_in.numel() < std::numeric_limits<int32_t>::max());
  auto t_in_flatten = t_in.flatten();
  auto t_out = at::empty_like(t_in_flatten);

  at::native::xpu::fbgemm_cumsum_kernel(t_out, t_in_flatten, 0);

  // make it exclusive
  t_out = t_out.roll(1, 0);
  // set all first elemnts 0
  t_out[0] = 0;
  return t_out.view_as(t_in);
}

// ============================================================================
// SYCL Kernel Implementations
// ============================================================================

// Kernel for permuting the lengths. Used for permutation of sparse features.
template <typename index_t>
SYCL_EXT_ONEAPI_FUNCTION_PROPERTY((syclexp::nd_range_kernel<1>))
void permute_2D_lengths_kernel_(
    int32_t T,
    int32_t B,
    const index_t* __restrict__ lengths,
    const int32_t* __restrict__ permute,
    index_t* __restrict__ permuted_lengths) {
  auto item = syclext::this_work_item::get_nd_item<1>();
  XPU_KERNEL_LOOP(item, b_t, B * T) {
    int32_t b = b_t % B;
    int32_t t = b_t / B;
    permuted_lengths[b_t] = lengths[permute[t] * B + b];
  }
}

void permute_2D_lengths_kernel_xpu(
    int32_t T,
    int32_t B,
    const at::Tensor& lengths_contig,
    const at::Tensor& permute_contig,
    at::Tensor& permuted_lengths) {
  constexpr int32_t threads_1 = 256;
  const auto blocks_1 = xpu_calc_xblock_count(B * T, threads_1);
  AT_DISPATCH_INDEX_TYPES(
      lengths_contig.scalar_type(), "permute_2D_lengths_kernel", [&] {
        sycl_kernel_submit<permute_2D_lengths_kernel_<index_t>>(
            sycl::range<1>(blocks_1 * threads_1),
            sycl::range<1>(threads_1),
            getCurrentSYCLQueue(),
            0,
            T,
            B,
            lengths_contig.data_ptr<index_t>(),
            permute_contig.data_ptr<int32_t>(),
            permuted_lengths.data_ptr<index_t>());
      });
}

template <
    bool has_weight,
    typename offsets_t,
    typename indices_t,
    typename weights_t>
SYCL_EXT_ONEAPI_FUNCTION_PROPERTY((syclexp::nd_range_kernel<2>))
void permute_2D_data_kernel_(
    int32_t len,
    int32_t T,
    int32_t B,
    const indices_t* __restrict__ indices,
    const weights_t* __restrict__ weights,
    const int32_t weights_columns,
    const int32_t* __restrict__ permute,
    const offsets_t* __restrict__ input_offsets,
    const offsets_t* __restrict__ output_offsets,
    indices_t* __restrict__ permuted_indices,
    weights_t* __restrict__ permuted_weights) {
  auto item = syclext::this_work_item::get_nd_item<2>();
  auto b_t_start =
      item.get_group(0) * item.get_local_range(1) + item.get_local_id(1);
  const auto stride = item.get_group_range(0) * item.get_local_range(1);
  for (int b_t = b_t_start; b_t < B * T; b_t += stride) {
    int32_t b = b_t % B;
    int32_t t = b_t / B;
    offsets_t output_start = output_offsets[b_t];
    offsets_t segment_length;
    if (b_t == B * T - 1) {
      segment_length = len - output_offsets[b_t];
    } else {
      segment_length = output_offsets[b_t + 1] - output_offsets[b_t];
    }
    offsets_t input_start = input_offsets[permute[t] * B + b];
    for (auto i = item.get_local_id(0); i < segment_length;
         i += item.get_local_range(0)) {
      permuted_indices[output_start + i] = indices[input_start + i];
      if (has_weight) {
        for (auto w_col = 0; w_col < weights_columns; ++w_col) {
          permuted_weights[(output_start + i) * weights_columns + w_col] =
              weights[(input_start + i) * weights_columns + w_col];
        }
      }
    }
  }
}

void permute_2D_data_kernel_xpu(
    int32_t permuted_indices_size,
    int32_t T,
    int32_t B,
    const at::Tensor& indices_contig,
    const std::optional<const at::Tensor>& weights,
    const int32_t weights_columns,
    const at::Tensor& permute_contig,
    const at::Tensor& input_offsets,
    const at::Tensor& output_offsets,
    at::Tensor& permuted_indices,
    const std::optional<at::Tensor>& permuted_weights) {
  constexpr int32_t BT_blocks = 32;
  const auto blocks_2 = xpu_calc_xblock_count(B * T, BT_blocks);
  AT_DISPATCH_INDEX_TYPES(
      input_offsets.scalar_type(), "permute_2D_data_kernel_1", [&] {
        using offsets_t = index_t;
        FBGEMM_DISPATCH_ALL_TYPES(
            indices_contig.scalar_type(), "permute_2D_data_kernel_2", [&] {
              using indices_t = scalar_t;
              if (weights.has_value()) {
                const auto weights_value_contig = weights.value().contiguous();
                FBGEMM_DISPATCH_ALL_TYPES_AND_DOUBLE(
                    weights_value_contig.scalar_type(),
                    "permute_2D_data_kernel_3",
                    [&] {
                      using weights_t = scalar_t;
                      sycl_kernel_submit<permute_2D_data_kernel_<
                          true,
                          offsets_t,
                          indices_t,
                          weights_t>>(
                          sycl::range<2>(blocks_2 * 32, BT_blocks),
                          sycl::range<2>(32, BT_blocks),
                          getCurrentSYCLQueue(),
                          0,
                          permuted_indices_size,
                          T,
                          B,
                          indices_contig.data_ptr<indices_t>(),
                          weights_value_contig.data_ptr<weights_t>(),
                          weights_columns,
                          permute_contig.data_ptr<int32_t>(),
                          input_offsets.data_ptr<offsets_t>(),
                          output_offsets.data_ptr<offsets_t>(),
                          permuted_indices.data_ptr<indices_t>(),
                          permuted_weights.value().data_ptr<weights_t>());
                    }); // for each weights_t
              } else {
                sycl_kernel_submit<permute_2D_data_kernel_<
                    false,
                    offsets_t,
                    indices_t,
                    float>>( // false float type here as wa since
                             // std::nullptr_t cannot be
                             // supported in free function
                             // kernel
                    sycl::range<2>(blocks_2 * 32, BT_blocks),
                    sycl::range<2>(32, BT_blocks),
                    getCurrentSYCLQueue(),
                    0,
                    permuted_indices_size,
                    T,
                    B,
                    indices_contig.data_ptr<indices_t>(),
                    nullptr,
                    0,
                    permute_contig.data_ptr<int32_t>(),
                    input_offsets.data_ptr<offsets_t>(),
                    output_offsets.data_ptr<offsets_t>(),
                    permuted_indices.data_ptr<indices_t>(),
                    nullptr);
              }
            }); // for each indices_t
      }); // for each offsets_t
}

} // namespace fbgemm_xpu
