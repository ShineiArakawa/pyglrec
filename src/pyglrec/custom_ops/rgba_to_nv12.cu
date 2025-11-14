#include <cuda_runtime.h>

#include <cstdint>
#include <rgba_to_nv12.cuh>
#include <utils.hpp>

__forceinline__ __device__ void rgb_to_yuv_601_full(
    float r, float g, float b,
    float& y, float& u, float& v) {
  y = 0.299f * r + 0.587f * g + 0.114f * b;
  u = -0.168736f * r - 0.331264f * g + 0.5f * b + 0.5f;
  v = 0.5f * r - 0.418688f * g - 0.081312f * b + 0.5f;
}

__forceinline__ __device__ std::uint8_t clamp8(float v) {
  v = fminf(fmaxf(v, 0.0f), 1.0f);  // clamp to [0.0, 1.0]
  return static_cast<std::uint8_t>(v * 255.0f + 0.5f);
}

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

  rgba_to_nv12_kernel<<<blocks_per_grid, threads_per_block>>>(
      src_rgba,
      pitch,
      width,
      height,
      dst_y,
      dst_uv);

  cudaErrors(cudaGetLastError());
}