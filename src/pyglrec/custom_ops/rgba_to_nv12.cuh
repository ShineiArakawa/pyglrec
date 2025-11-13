#pragma once

// --------------------------------------------------------------------------------------------------------------------------------
// Utilities

static int div_up(int a, int b) {
  return (a + b - 1) / b;
}

// --------------------------------------------------------------------------------------------------------------------------------
// Host function declarations

void rgba_to_nv12(uint64_t src_rgba_ptr,  // Linear RGBA cuda memory pointer
                  uint64_t src_pitch,     // Linear RGBA cuda memory pitch (from cudaMallocPitch)
                  int width,              // Width of the frame
                  int height,             // Height of the frame
                  uint64_t dst_nv12       // d_ptr (width * height * 3 / 2 bytes)
);
