// Copyright (c) 2026 Intel Corporation. All Rights Reserved.

#ifdef WITH_SYCL_KERNELS

#include "ColorConversionKernel.h"
#include <algorithm> // For std::clamp

namespace facebook::torchcodec {

using float3x3 = std::array<sycl::float3, 3>;

const float3x3 rgb_matrix_bt709 = {
  sycl::float3{ 1.0, 0.0, 1.5748 },
  sycl::float3{ 1.0, -0.187324, -0.468124 },
  sycl::float3{ 1.0, 1.8556, 0.0 }
};

//const sycl::float3 rgb_matrix_bt601[3] = {
//  { 1.0, 0.0, 1.402 },
//  { 1.0, -0.344136, -0.714136 },
//  { 1.0, 1.772, 0.0}
//};

// Helper function for the Intel Tile-Y offset calculation
// Intel Y-Tiling uses COLUMN-MAJOR OWord (16 bytes) organization
// Tile: 128 bytes wide × 32 rows = 4KB
// Within tile: 8 OWords (16-byte columns) arranged column-by-column
// Each OWord covers all 32 rows before moving to next OWord
size_t get_tile_offset(int x, int y, int stride) {
  const int TileW = 128;  // Tile width in bytes
  const int TileH = 32;   // Tile height in rows
  const int OWordSize = 16; // OWord = 16 bytes
  const int TileSize = TileW * TileH;  // 4096 bytes per tile

  // Which tile does this pixel belong to?
  int tile_x = x / TileW;
  int tile_y = y / TileH;

  // Position within the tile
  int x_in_tile = x % TileW;
  int y_in_tile = y % TileH;

  // Block position added to remove swap of 64-byte blocks in the tile (TileY XOR pattern)
  int block_x = x_in_tile / 64;  // width of pixel blocks
  int block_y = y_in_tile / 4;   // heigh of pixel blocks

  // Y-Tiling: Column-major OWord layout
  // OWord index (0-7): which 16-byte column within the tile
  int oword_idx = x_in_tile / OWordSize;
  // Offset within OWord (0-15)
  int offset_in_oword = x_in_tile % OWordSize;

  int sub_tile_size = OWordSize * 4;
  int sub_tile_y = y_in_tile / 4;
  int y_in_sub_tile = y_in_tile % 4;

  // conditional to remove swap of 64-byte blocks in the tile (TileY XOR pattern)
  if ((block_x ^ block_y ) & 0x1){
    block_x ^= 1;
    block_y ^= 1;

    x_in_tile = block_x * 64 + (x_in_tile % 64);
    y_in_tile = block_y * 4 + (y_in_tile % 4);

    sub_tile_y = block_y;
    y_in_sub_tile = y_in_tile % 4;

    oword_idx = x_in_tile / OWordSize;
    offset_in_oword = x_in_tile % 16;
  }

  int offset_in_tile = (sub_tile_y * TileW/OWordSize + oword_idx) * sub_tile_size + y_in_sub_tile * OWordSize + offset_in_oword;

  // Number of tiles per row
  int stride_in_tiles = stride / TileW;

  // Final tiled offset
  size_t tile_offset = (size_t)(tile_y * stride_in_tiles + tile_x) * TileSize;
  return tile_offset + offset_in_tile;
}

sycl::uchar3 yuv2rgb(uint8_t y, uint8_t u, uint8_t v, bool fullrange, const float3x3 &rgb_matrix) {
  sycl::float3 src;
  if (fullrange) {
    src = sycl::float3(y/255.0f, (u-128.0f)/255.0f - 0.5f, (v-128.0f)/255.0f - 0.5f);
  } else {
    src = sycl::float3((y-16.0f)/219.0f, (u-128.0f)/224.0f, (v-128.0f)/224.0f);
  }

  sycl::float3 fdst;
  fdst.x() = sycl::dot(src, rgb_matrix[0]);
  fdst.y() = sycl::dot(src, rgb_matrix[1]);
  fdst.z() = sycl::dot(src, rgb_matrix[2]);

  sycl::uchar3 dst;
  dst.x() = (uint8_t)std::clamp(fdst[0] * 255.0f, 0.0f, 255.0f);
  dst.y() = (uint8_t)std::clamp(fdst[1] * 255.0f, 0.0f, 255.0f);
  dst.z() = (uint8_t)std::clamp(fdst[2] * 255.0f, 0.0f, 255.0f);
  return dst;
}

struct NV12toRGBKernel {
  const uint8_t* y_plane;
  const uint8_t* uv_plane;
  uint8_t* rgb_output;
  int width;
  int height;
  int stride;
  bool fullrange;
  float3x3 rgb_matrix;

  NV12toRGBKernel(
      const uint8_t* y_plane,
      const uint8_t* uv_plane,
      uint8_t* rgb_output,
      int width,
      int height,
      int stride,
      bool fullrange,
      const float3x3 &rgb_matrix):
    y_plane(y_plane),
    uv_plane(uv_plane),
    rgb_output(rgb_output),
    width(width),
    height(height),
    stride(stride),
    fullrange(fullrange),
    rgb_matrix(rgb_matrix)
  {}

  void operator()(sycl::id<2> idx) const {
    int yx = idx[1];
    int yy = idx[0];

    if (yx >= width || yy >= height) {
      return;
    }

    int ux = sycl::floor(yx/2.0);
    int uy = sycl::floor(yy/2.0);

    size_t tiled_idx_y = get_tile_offset(yx, yy, stride);
    size_t tiled_idx_u = get_tile_offset(2*ux, uy, stride);
    size_t tiled_idx_v = get_tile_offset(2*ux+1, uy, stride);

    uint8_t y = y_plane[tiled_idx_y];
    uint8_t u = uv_plane[tiled_idx_u];
    uint8_t v = uv_plane[tiled_idx_v];

    sycl::uchar3 rgb = yuv2rgb(y, u, v, fullrange, rgb_matrix);

    int rgb_idx = 3 * (yy * width + yx);

    rgb_output[rgb_idx + 0] = rgb.x();
    rgb_output[rgb_idx + 1] = rgb.y();
    rgb_output[rgb_idx + 2] = rgb.z();
  }
};

void convertNV12ToRGB(
    sycl::queue& queue,
    const uint8_t* y_plane,
    const uint8_t* uv_plane,
    uint8_t* rgb_output,
    int width,
    int height,
    int stride,
    bool fullrange) {
  queue.submit([&](sycl::handler& cgh) {
    NV12toRGBKernel kernel(
      y_plane, uv_plane, rgb_output,
      width, height, stride,
      fullrange, rgb_matrix_bt709);

    cgh.parallel_for(
        sycl::range<2>(height, width),
        kernel);
  });

  queue.wait();
}

// This function is called during library initialization to ensure
// the SYCL runtime registers the kernel associated with this type.
void registerColorConversionKernel() {
  // Creating a dummy pointer to the kernel type is often enough
  // to force the compiler to emit the necessary RTTI/integration info.
  // We use volatile to prevent optimization.
  volatile size_t s = sizeof(NV12toRGBKernel);
  (void)s;
}

} // namespace facebook::torchcodec
#endif // WITH_SYCL_KERNELS
