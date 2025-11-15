#include <cuda_runtime.h>

#include <cstdint>
#include <rgba_conversion.cuh>
#include <utils.hpp>

// --------------------------------------------------------------------------------------------------------------------------------
// Device functions and kernels

/**
 * @brief Convert RGB color to YUV color space using BT.601 full range conversion.
 *
 * @param r Red component in [0.0, 1.0]
 * @param g Green component in [0.0, 1.0]
 * @param b Blue component in [0.0, 1.0]
 * @param y Luminance component in [0.0, 1.0]
 * @param u Chrominance component U in [0.0, 1.0]
 * @param v Chrominance component V in [0.0, 1.0]
 *
 * @note See also: https://ja.wikipedia.org/wiki/YUV
 */
__forceinline__ __device__ void rgb_to_yuv_601_full(float r, float g, float b,
                                                    float& y, float& u, float& v) {
  // clang-format off
  y =   0.299f    * r + 0.587f    * g + 0.114f    * b;         // in [0.0, 1.0]
  u = - 0.168736f * r - 0.331264f * g + 0.5f      * b + 0.5f;  // in [0.0, 1.0]
  v =   0.5f      * r - 0.418688f * g - 0.081312f * b + 0.5f;  // in [0.0, 1.0]
  // clang-format on
}

/**
 * @brief Convert YUV color to RGB color space using BT.601 full range conversion.
 *
 * @param y Luminance component in [0.0, 1.0]
 * @param u Chrominance component U in [0.0, 1.0]
 * @param v Chrominance component V in [0.0, 1.0]
 * @param r Red component in [0.0, 1.0]
 * @param g Green component in [0.0, 1.0]
 * @param b Blue component in [0.0, 1.0]
 *
 * @note See also: https://ja.wikipedia.org/wiki/YUV
 */
__forceinline__ __device__ void yuv_to_rgb_601_full(float y, float u, float v,
                                                    float& r, float& g, float& b) {
  u -= 0.5f;  // in [-0.5, 0.5]
  v -= 0.5f;  // in [-0.5, 0.5]

  // clang-format off
  r = 1.0f * y + 0.0f      * u + 1.402f    * v; // in [0.0, 1.0]
  g = 1.0f * y - 0.344136f * u - 0.714136f * v; // in [0.0, 1.0]
  b = 1.0f * y + 1.772f    * u + 0.0f      * v; // in [0.0, 1.0]
  // clang-format on
}

/**
 * @brief Clamp a float value to the range [0, 255] and convert to uint8_t.
 *
 * @param v
 */
__forceinline__ __device__ std::uint8_t clamp8(float v) {
  v = fminf(fmaxf(v, 0.0f), 1.0f);  // clamp to [0.0, 1.0]
  return static_cast<std::uint8_t>(v * 255.0f + 0.5f);
}

/**
 * @brief Convert RGBA image to NV12 format. This kernel also flips the image vertically during the conversion.
 *
 * @param src_rgba
 * @param src_pitch
 * @param width
 * @param height
 * @param dst_y
 * @param dst_uv
 */
__global__ void rgba_to_nv12_kernel(
    const uchar4* __restrict__ src_rgba,
    size_t src_pitch,  // bytes per row
    int width,
    int height,
    std::uint8_t* __restrict__ dst_y,
    std::uint8_t* __restrict__ dst_uv) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;  // pixel x
  int y = blockIdx.y * blockDim.y + threadIdx.y;  // pixel y

  if (x >= width || y >= height) return;

  // First element in the row
  const std::uint8_t* row_base = reinterpret_cast<const std::uint8_t*>(src_rgba) + (height - 1 - y) * src_pitch;
  const uchar4* row = reinterpret_cast<const uchar4*>(row_base);

  // RGBA pixel at (x, y)
  uchar4 p = row[x];

  float r = p.x / 255.0f;
  float g = p.y / 255.0f;
  float b = p.z / 255.0f;

  float Yf, Uf, Vf;
  rgb_to_yuv_601_full(r, g, b, Yf, Uf, Vf);

  int idx = y * width + x;
  dst_y[idx] = clamp8(Yf);  // First element is Y

  // For UV plane, one sample per 2x2 block
  if ((x % 2 == 0) && (y % 2 == 0)) {
    float Uacc = 0.0f;
    float Vacc = 0.0f;
    int count = 0;

    for (int dy = 0; dy < 2; ++dy) {
      int yy = y + dy;
      if (yy >= height) {
        continue;
      }

      const std::uint8_t* row_base2 = reinterpret_cast<const std::uint8_t*>(src_rgba) + (height - 1 - yy) * src_pitch;
      const uchar4* row2 = reinterpret_cast<const uchar4*>(row_base2);

      for (int dx = 0; dx < 2; ++dx) {
        int xx = x + dx;
        if (xx >= width) {
          continue;
        }

        uchar4 p2 = row2[xx];
        float r2 = p2.x / 255.0f;
        float g2 = p2.y / 255.0f;
        float b2 = p2.z / 255.0f;

        float Y2, U2, V2;
        rgb_to_yuv_601_full(r2, g2, b2, Y2, U2, V2);

        Uacc += U2;
        Vacc += V2;
        ++count;
      }
    }

    if (count > 0) {
      float Uavg = Uacc / count;
      float Vavg = Vacc / count;

      std::uint8_t U8 = clamp8(Uavg);
      std::uint8_t V8 = clamp8(Vavg);

      // UV index
      int uv_row = y / 2;
      int uv_col = x;
      int uv_idx = uv_row * width + uv_col;

      dst_uv[uv_idx + 0] = U8;
      dst_uv[uv_idx + 1] = V8;
    }
  }
}

/**
 * @brief Convert NV12 image to RGBA format. This kernel also flips the image vertically during the conversion.
 * Be careful that the NV12 format has 4:2:0 chroma subsampling, therefore its RGBA conversion always loses some color information.
 *
 * @param src_nv12
 * @param width
 * @param height
 * @param dst_rgba
 * @param dst_pitch
 */
__global__ void nv12_to_rgba_kernel(const std::uint8_t* __restrict__ src_nv12,
                                    int width,
                                    int height,
                                    uchar4* __restrict__ dst_rgba,
                                    size_t dst_pitch  // bytes per row
) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;  // pixel x at the output RGBA image
  int y = blockIdx.y * blockDim.y + threadIdx.y;  // pixel y at the output RGBA image

  if (x >= width || y >= height) return;

  // Get Y value
  int y_idx = y * width + x;
  float Y = static_cast<float>(src_nv12[y_idx]) / 255.0f;  // in [0.0, 1.0]

  // Get U and V values

  int uv_width = width / 2;

  int uv_x = x / 2;
  int uv_y = y / 2;

  int u_offset = width * height;
  int v_offset = u_offset + (width * height) / 4;
  int loc = uv_y * uv_width + uv_x;

  float U = static_cast<float>(src_nv12[u_offset + loc]) / 255.0f;  // in [0.0, 1.0]
  float V = static_cast<float>(src_nv12[v_offset + loc]) / 255.0f;  // in [0.0, 1.0]

  // Convert YUV to RGB
  float R, G, B;
  yuv_to_rgb_601_full(Y, U, V, R, G, B);

  std::uint8_t R8 = clamp8(R);
  std::uint8_t G8 = clamp8(G);
  std::uint8_t B8 = clamp8(B);

  // Write to RGBA output (flipped vertically)
  std::uint8_t* row_base = reinterpret_cast<std::uint8_t*>(dst_rgba) + (height - 1 - y) * dst_pitch;
  uchar4* row = reinterpret_cast<uchar4*>(row_base);
  row[x] = make_uchar4(R8, G8, B8, 255);
}

/**
 * @brief Convert RGBA image to YUV444 format. This kernel also flips the image vertically during the conversion.
 *
 * @param src_rgba
 * @param src_pitch
 * @param width
 * @param height
 * @param dst_yuv
 */
__global__ void rgba_to_yuv444_kernel(const uchar4* __restrict__ src_rgba,
                                      size_t src_pitch,  // bytes per row
                                      int width,
                                      int height,
                                      std::uint8_t* __restrict__ dst_yuv) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;  // pixel x
  int y = blockIdx.y * blockDim.y + threadIdx.y;  // pixel

  if (x >= width || y >= height) return;

  // First element in the row
  const std::uint8_t* row_base = reinterpret_cast<const std::uint8_t*>(src_rgba) + (height - 1 - y) * src_pitch;
  const uchar4* row = reinterpret_cast<const uchar4*>(row_base);

  const uchar4 rgba = row[x];

  float r = rgba.x / 255.0f;
  float g = rgba.y / 255.0f;
  float b = rgba.z / 255.0f;

  float Yf, Uf, Vf;
  rgb_to_yuv_601_full(r, g, b, Yf, Uf, Vf);

  int idx = y * width + x;
  int pitch = width * height;
  dst_yuv[idx + 0 * pitch] = clamp8(Yf);  // Y
  dst_yuv[idx + 1 * pitch] = clamp8(Uf);  // U
  dst_yuv[idx + 2 * pitch] = clamp8(Vf);  // V
}

/**
 * @brief Convert YUV444 image to RGBA format. This kernel also flips the image vertically during the conversion.
 *
 * @param src_yuv
 * @param width
 * @param height
 * @param dst_rgba
 * @param dst_pitch
 */
__global__ void yuv444_to_rgba_kernel(std::uint8_t* __restrict__ src_yuv,
                                      int width,
                                      int height,
                                      uchar4* __restrict__ dst_rgba,
                                      size_t dst_pitch  // bytes per row
) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;  // pixel x
  int y = blockIdx.y * blockDim.y + threadIdx.y;  // pixel y

  if (x >= width || y >= height) return;

  int idx = y * width + x;
  int pitch = width * height;

  float Y = static_cast<float>(src_yuv[idx + 0 * pitch]) / 255.0f;  // in [0.0, 1.0]
  float U = static_cast<float>(src_yuv[idx + 1 * pitch]) / 255.0f;  // in [0.0, 1.0]
  float V = static_cast<float>(src_yuv[idx + 2 * pitch]) / 255.0f;  // in [0.0, 1.0]

  float R, G, B;
  yuv_to_rgb_601_full(Y, U, V, R, G, B);

  std::uint8_t R8 = clamp8(R);
  std::uint8_t G8 = clamp8(G);
  std::uint8_t B8 = clamp8(B);

  // Write to RGBA output (flipped vertically)
  std::uint8_t* row_base = reinterpret_cast<std::uint8_t*>(dst_rgba) + (height - 1 - y) * dst_pitch;
  uchar4* row = reinterpret_cast<uchar4*>(row_base);
  row[x] = make_uchar4(R8, G8, B8, 255);
}

// --------------------------------------------------------------------------------------------------------------------------------
// Host functions

void rgba_to_nv12(uint64_t src_rgba_ptr,  // Linear RGBA cuda memory pointer
                  uint64_t src_pitch,     // Linear RGBA cuda memory pitch (from cudaMallocPitch)
                  int width,              // Width of the frame
                  int height,             // Height of the frame
                  uint64_t dst_nv12       // d_ptr (width * height * 3 / 2 bytes)
) {
  const uchar4* src_rgba = reinterpret_cast<const uchar4*>(src_rgba_ptr);
  size_t pitch = static_cast<size_t>(src_pitch);

  auto* dst_base = reinterpret_cast<std::uint8_t*>(dst_nv12);
  std::uint8_t* dst_y = dst_base;
  std::uint8_t* dst_uv = dst_base + width * height;

  dim3 threads_per_block(32, 16);
  dim3 blocks_per_grid(div_up(width, threads_per_block.x),
                       div_up(height, threads_per_block.y));

  rgba_to_nv12_kernel<<<blocks_per_grid, threads_per_block>>>(src_rgba,
                                                              pitch,
                                                              width,
                                                              height,
                                                              dst_y,
                                                              dst_uv);

  cudaErrors(cudaGetLastError());
}

void nv12_to_rgba(uint64_t src_nv12_ptr,  // Linear NV12 cuda memory pointer
                  int width,              // Width of the frame
                  int height,             // Height of the frame
                  uint64_t dst_rgba_ptr,  // Linear RGBA cuda memory pointer
                  uint64_t dst_pitch      // Linear RGBA cuda memory pitch (from cudaMallocPitch)
) {
  const std::uint8_t* src_nv12 = reinterpret_cast<const std::uint8_t*>(src_nv12_ptr);
  uchar4* dst_rgba = reinterpret_cast<uchar4*>(dst_rgba_ptr);
  size_t pitch = static_cast<size_t>(dst_pitch);

  dim3 threads_per_block(32, 16);
  dim3 blocks_per_grid(div_up(width, threads_per_block.x),
                       div_up(height, threads_per_block.y));

  nv12_to_rgba_kernel<<<blocks_per_grid, threads_per_block>>>(src_nv12,
                                                              width,
                                                              height,
                                                              dst_rgba,
                                                              pitch);

  cudaErrors(cudaGetLastError());
}

void rgba_to_yuv444(uint64_t src_rgba_ptr,  // Linear RGBA cuda memory pointer
                    uint64_t src_pitch,     // Linear RGBA cuda memory pitch (from cudaMallocPitch)
                    int width,              // Width of the frame
                    int height,             // Height of the frame
                    uint64_t dst_yuv        // d_ptr (width * height * 3 bytes)
) {
  const uchar4* src_rgba = reinterpret_cast<const uchar4*>(src_rgba_ptr);
  size_t pitch = static_cast<size_t>(src_pitch);

  std::uint8_t* dst_yuv444 = reinterpret_cast<std::uint8_t*>(dst_yuv);

  dim3 threads_per_block(32, 16);
  dim3 blocks_per_grid(div_up(width, threads_per_block.x),
                       div_up(height, threads_per_block.y));

  rgba_to_yuv444_kernel<<<blocks_per_grid, threads_per_block>>>(src_rgba,
                                                                pitch,
                                                                width,
                                                                height,
                                                                dst_yuv444);

  cudaErrors(cudaGetLastError());
}

void yuv444_to_rgba(uint64_t src_yuv_ptr,   // Linear YUV cuda memory pointer
                    int width,              // Width of the frame
                    int height,             // Height of the frame
                    uint64_t dst_rgba_ptr,  // Linear RGBA cuda memory pointer
                    uint64_t dst_pitch      // Linear RGBA cuda memory pitch (from cudaMallocPitch)
) {
  std::uint8_t* src_yuv444 = reinterpret_cast<std::uint8_t*>(src_yuv_ptr);
  uchar4* dst_rgba = reinterpret_cast<uchar4*>(dst_rgba_ptr);
  size_t pitch = static_cast<size_t>(dst_pitch);

  dim3 threads_per_block(32, 16);
  dim3 blocks_per_grid(div_up(width, threads_per_block.x),
                       div_up(height, threads_per_block.y));

  yuv444_to_rgba_kernel<<<blocks_per_grid, threads_per_block>>>(src_yuv444,
                                                                width,
                                                                height,
                                                                dst_rgba,
                                                                pitch);

  cudaErrors(cudaGetLastError());
}
