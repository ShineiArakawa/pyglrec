#pragma once

// --------------------------------------------------------------------------------------------------------------------------------
// Utilities

static int div_up(int a, int b) {
  return (a + b - 1) / b;
}

// --------------------------------------------------------------------------------------------------------------------------------
// Host function declarations

/**
 * @brief Convert RGBA image to NV12 format with 4:2:0 chroma subsampling.
 * This function also flips the image vertically during the conversionß
 *
 * @param src_rgba_ptr
 * @param src_pitch
 * @param width
 * @param height
 * @param dst_nv12
 */
void rgba_to_nv12(uint64_t src_rgba_ptr,  // Linear RGBA cuda memory pointer
                  uint64_t src_pitch,     // Linear RGBA cuda memory pitch (from cudaMallocPitch)
                  int width,              // Width of the frame
                  int height,             // Height of the frame
                  uint64_t dst_nv12       // d_ptr (width * height * 3 / 2 bytes)
);

/**
 * @brief Convert NV12 image to RGBA format. This function also flips the image vertically during the conversion.
 *
 * @param src_nv12_ptr
 * @param width
 * @param height
 * @param dst_rgba_ptr
 * @param dst_pitch
 */
void nv12_to_rgba(uint64_t src_nv12_ptr,  // Linear NV12 cuda memory pointer
                  int width,              // Width of the frame
                  int height,             // Height of the frame
                  uint64_t dst_rgba_ptr,  // Linear RGBA cuda memory pointer
                  uint64_t dst_pitch      // Linear RGBA cuda memory pitch (from cudaMallocPitch)
);

/**
 * @brief Convert RGBA image to YUV lossless 4:4:4 format. This function does not perform any vertical flipping.
 *
 * @param src_rgba_ptr
 * @param src_pitch
 * @param width
 * @param height
 * @param dst_yuv
 */
void rgba_to_yuv444(uint64_t src_rgba_ptr,  // Linear RGBA cuda memory pointer
                    uint64_t src_pitch,     // Linear RGBA cuda memory pitch (from cudaMallocPitch)
                    int width,              // Width of the frame
                    int height,             // Height of the frame
                    uint64_t dst_yuv        // d_ptr (width * height * 3 bytes)
);

/**
 * @brief Convert YUV lossless 4:4:4 image to RGBA format. This function does not perform any vertical flipping.
 *
 * @param src_yuv_ptr
 * @param width
 * @param height
 * @param dst_rgba_ptr
 * @param dst_pitch
 */
void yuv444_to_rgba(uint64_t src_yuv_ptr,   // Linear YUV cuda memory pointer
                    int width,              // Width of the frame
                    int height,             // Height of the frame
                    uint64_t dst_rgba_ptr,  // Linear RGBA cuda memory pointer
                    uint64_t dst_pitch      // Linear RGBA cuda memory pitch (from cudaMallocPitch)
);