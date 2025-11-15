/**
 * @file cuda_gl_interop.cpp
 * @author Shinei Arakawa (sarakawalab@gmail.com)
 * @brief CUDA-OpenGL inter-operability functions
 */

#include <cuda_runtime.h>

#ifdef WIN32
// clang-format off
#include <windows.h>
#include <gl/GL.h>
// clang-format on
#else
#include <GL/gl.h>
#endif

#include <cuda_gl_interop.h>
#include <pybind11/pybind11.h>

#include <iostream>
#include <rgba_conversion.cuh>
#include <sstream>
#include <string>
#include <tuple>
#include <utils.hpp>

// --------------------------------------------------------------------------------------------------------------------------------
// OpenGL API Call

/**
 * @brief Set the cuda device for current OpenGL context object
 *
 * @param device_id
 */
void set_cuda_device_for_current_OpenGL_context(int device_id) {
  cudaErrors(cudaSetDevice(device_id));
}

/**
 * @brief Get the cuda device ID for current OpenGL context object
 *
 * @return int CUDA device ID
 */
int get_cuda_device_for_current_OpenGL_context() {
  unsigned int device_count;
  int devices[16] = {};  // Assume a maximum of 16 devices

  cudaErrors(cudaGLGetDevices(&device_count,
                              &devices[0],
                              16,
                              cudaGLDeviceListAll));

  if (device_count == 0) {
    throw std::runtime_error("cudaGLGetDevices returned no devices for the current OpenGL context.");
  }

  int selected_device = devices[0];

  const GLubyte* renderer_ptr = glGetString(GL_RENDERER);
  std::string renderer = renderer_ptr ? reinterpret_cast<const char*>(renderer_ptr) : "";

  if (!renderer.empty()) {
    for (unsigned int i = 0; i < device_count; ++i) {
      cudaDeviceProp prop{};
      cudaErrors(cudaGetDeviceProperties(&prop, devices[i]));
      if (renderer.find(prop.name) != std::string::npos) {
        selected_device = devices[i];
        break;
      }
    }
  }

  return selected_device;
}

// --------------------------------------------------------------------------------------------------------------------------------
// Device memory management

// Linear 1D device memory allocation and deallocation

/**
 * @brief Allocate 1D device memory
 *
 * @param size_in_bytes Size in bytes
 * @return uint64_t Pointer to allocated device memory (as uint64_t)
 */
uint64_t allocate_device_memory_1d(size_t size_in_bytes) {
  void* ptr = nullptr;
  cudaErrors(cudaMalloc(&ptr, size_in_bytes));
  cudaErrors(cudaDeviceSynchronize());
  return reinterpret_cast<uint64_t>(ptr);
}

// Linear 2D device memory allocation

/**
 * @brief Allocate 2D device memory
 *
 * @param width Width in elements
 * @param height Height in elements
 * @param element_size Size of each element in bytes
 * @return std::tuple<uint64_t, uint64_t> Tuple of (pointer to allocated device memory as uint64_t, pitch in bytes)
 */
std::tuple<uint64_t, uint64_t> allocate_device_memory_2d(int width, int height, uint64_t element_size) {
  void* d_ptr = nullptr;
  size_t pitch = 0;
  cudaErrors(cudaMallocPitch(reinterpret_cast<void**>(&d_ptr),
                             &pitch,
                             static_cast<size_t>(width) * element_size,
                             static_cast<size_t>(height)));
  cudaErrors(cudaDeviceSynchronize());
  return std::make_tuple(reinterpret_cast<uint64_t>(d_ptr), static_cast<uint64_t>(pitch));
}

/**
 * @brief Free device memory
 *
 * @param d_ptr Pointer to device memory (as uint64_t)
 */
void free_device_memory(uint64_t d_ptr) {
  if (d_ptr == 0) {
    return;
  }
  cudaErrors(cudaFree(reinterpret_cast<void*>(d_ptr)));
  cudaErrors(cudaDeviceSynchronize());
}

// --------------------------------------------------------------------------------------------------------------------------------
// Texture copy with YUV conversion

/**
 * @brief Copy an OpenGL texture to CUDA device memory
 *
 * @param texture_id OpenGL texture ID
 * @param width Width of the texture
 * @param height Height of the texture
 * @param cuda_tex_ptr Pointer to CUDA device memory
 * @param cuda_tex_pitch Pitch of the CUDA device memory
 */
void copy_texture_to_cuda_memory(uint64_t texture_id,
                                 int width,
                                 int height,
                                 uint64_t cuda_tex_ptr,
                                 uint64_t cuda_tex_pitch) {
  // Register the texture with CUDA
  cudaGraphicsResource* cuda_resource = nullptr;
  cudaErrors(cudaGraphicsGLRegisterImage(&cuda_resource,
                                         static_cast<GLuint>(texture_id),
                                         GL_TEXTURE_2D,
                                         cudaGraphicsRegisterFlagsReadOnly));

  // Map the resource
  cudaErrors(cudaGraphicsMapResources(1, &cuda_resource, 0));

  // Get the mapped array
  cudaArray_t cuda_array = nullptr;
  cudaErrors(cudaGraphicsSubResourceGetMappedArray(&cuda_array, cuda_resource, 0, 0));

  // Copy the texture data to linear CUDA memory (RGBA8 / uchar4)
  const size_t row_bytes = static_cast<size_t>(width) * sizeof(uchar4);
  const size_t dst_pitch = static_cast<size_t>(cuda_tex_pitch);

  cudaErrors(cudaMemcpy2DFromArray(reinterpret_cast<void*>(cuda_tex_ptr),
                                   dst_pitch,
                                   cuda_array,
                                   0,
                                   0,
                                   row_bytes,
                                   height,
                                   cudaMemcpyDeviceToDevice));

  // Unmap the resource
  cudaErrors(cudaGraphicsUnmapResources(1, &cuda_resource, 0));
  cudaErrors(cudaGraphicsUnregisterResource(cuda_resource));

  // Synchronize device
  cudaErrors(cudaDeviceSynchronize());
}

/**
 * @brief Convert RGBA texture to NV12 format, which involves chroma subsampling with 4:2:0
 *
 * @param width Width of the texture
 * @param height Height of the texture
 * @param cuda_tex_ptr Pointer to CUDA device memory containing the RGBA texture
 * @param cuda_tex_pitch Pitch of the CUDA device memory containing the RGBA texture
 * @param d_ptr Pointer to CUDA device memory for the NV12 output
 */
void convert_rgba_to_nv12(int width,
                          int height,
                          uint64_t cuda_tex_ptr,
                          uint64_t cuda_tex_pitch,
                          uint64_t d_ptr) {
  // Copy and convert RGBA to NV12
  rgba_to_nv12(cuda_tex_ptr,
               cuda_tex_pitch,
               width,
               height,
               d_ptr);

  // Synchronize device
  cudaErrors(cudaDeviceSynchronize());
}

/**
 * @brief Convert NV12 format to RGBA texture
 *
 * @param width Width of the texture
 * @param height Height of the texture
 * @param cuda_nv12_ptr Pointer to CUDA device memory containing the NV12 texture
 * @param d_ptr Pointer to CUDA device memory for the RGBA output
 * @param d_pitch Pitch of the CUDA device memory for the RGBA output
 */
void convert_nv12_to_rgba(int width,
                          int height,
                          uint64_t cuda_nv12_ptr,
                          uint64_t d_ptr,
                          uint64_t d_pitch) {
  // Copy and convert NV12 to RGBA
  nv12_to_rgba(cuda_nv12_ptr,
               width,
               height,
               d_ptr,
               d_pitch);

  // Synchronize device
  cudaErrors(cudaDeviceSynchronize());
}

/**
 * @brief Convert RGBA texture to YUV444 format, which is a lossless format without chroma subsampling
 *
 * @param width Width of the texture
 * @param height Height of the texture
 * @param cuda_tex_ptr Pointer to CUDA device memory containing the RGBA texture
 * @param cuda_tex_pitch Pitch of the CUDA device memory containing the RGBA texture
 * @param d_ptr Pointer to CUDA device memory for the YUV444 output
 */
void convert_rgba_to_yuv444(int width,
                            int height,
                            uint64_t cuda_tex_ptr,
                            uint64_t cuda_tex_pitch,
                            uint64_t d_ptr) {
  // Copy and convert RGBA to YUV444
  rgba_to_yuv444(cuda_tex_ptr,
                 cuda_tex_pitch,
                 width,
                 height,
                 d_ptr);

  // Synchronize device
  cudaErrors(cudaDeviceSynchronize());
}

/**
 * @brief Convert YUV444 format to RGBA texture
 *
 * @param width Width of the texture
 * @param height Height of the texture
 * @param cuda_yuv444_ptr Pointer to CUDA device memory containing the YUV444 texture
 * @param d_ptr Pointer to CUDA device memory for the RGBA output
 * @param d_pitch Pitch of the CUDA device memory for the RGBA output
 */
void convert_yuv444_to_rgba(int width,
                            int height,
                            uint64_t cuda_yuv444_ptr,
                            uint64_t d_ptr,
                            uint64_t d_pitch) {
  // Copy and convert YUV444 to RGBA
  yuv444_to_rgba(cuda_yuv444_ptr,
                 width,
                 height,
                 d_ptr,
                 d_pitch);

  // Synchronize device
  cudaErrors(cudaDeviceSynchronize());
}

// --------------------------------------------------------------------------------------------------------------------------------
// Python bindings

PYBIND11_MODULE(cuda_gl_interop, m) {
  // Generate documentation
  std::stringstream doc_stream;
  doc_stream << "A plugin for CUDA-OpenGL inter-operability functionalities." << std::endl;
  doc_stream << std::endl;
  doc_stream << "(Compiled on " << __DATE__ << " at " << __TIME__ << ", with ";
#if defined(__clang__)
  doc_stream << "Clang version " << __clang_major__ << "." << __clang_minor__;
#elif defined(__GNUC__)
  doc_stream << "GCC version " << __GNUC__ << "." << __GNUC_MINOR__;
#elif defined(_MSC_VER)
  doc_stream << "MSVC version " << _MSC_VER;
#else
  doc_stream << "Unknown compiler";
#endif
  doc_stream << ")" << std::endl;
  doc_stream << std::endl;

  m.doc() = doc_stream.str().c_str();

  m.def("allocate_device_memory_1d",
        &allocate_device_memory_1d,
        "Allocate device memory of given size (in bytes)",
        pybind11::arg("size_in_bytes"));
  m.def("allocate_device_memory_2d",
        &allocate_device_memory_2d,
        "Allocate 2D device memory",
        pybind11::arg("width"),
        pybind11::arg("height"),
        pybind11::arg("element_size"));
  m.def("free_device_memory",
        &free_device_memory,
        "Free device memory",
        pybind11::arg("ptr"));
  m.def("copy_texture_to_cuda_memory",
        &copy_texture_to_cuda_memory,
        "Copy OpenGL texture to CUDA device linear memory",
        pybind11::arg("texture_id"),
        pybind11::arg("width"),
        pybind11::arg("height"),
        pybind11::arg("cuda_tex_ptr"),
        pybind11::arg("cuda_tex_pitch"));
  m.def("convert_rgba_to_nv12",
        &convert_rgba_to_nv12,
        "Convert RGBA texture in CUDA linear memory to NV12 format in CUDA memory",
        pybind11::arg("width"),
        pybind11::arg("height"),
        pybind11::arg("cuda_tex_ptr"),
        pybind11::arg("cuda_tex_pitch"),
        pybind11::arg("d_ptr"));
  m.def("convert_nv12_to_rgba",
        &convert_nv12_to_rgba,
        "Convert NV12 format in CUDA memory to RGBA texture in CUDA linear memory",
        pybind11::arg("width"),
        pybind11::arg("height"),
        pybind11::arg("cuda_nv12_ptr"),
        pybind11::arg("d_ptr"),
        pybind11::arg("d_pitch"));
  m.def("convert_rgba_to_yuv444",
        &convert_rgba_to_yuv444,
        "Convert RGBA texture in CUDA linear memory to YUV444 format in CUDA memory",
        pybind11::arg("width"),
        pybind11::arg("height"),
        pybind11::arg("cuda_tex_ptr"),
        pybind11::arg("cuda_tex_pitch"),
        pybind11::arg("d_ptr"));
  m.def("convert_yuv444_to_rgba",
        &convert_yuv444_to_rgba,
        "Convert YUV444 format in CUDA memory to RGBA texture in CUDA linear memory",
        pybind11::arg("width"),
        pybind11::arg("height"),
        pybind11::arg("cuda_yuv444_ptr"),
        pybind11::arg("d_ptr"),
        pybind11::arg("d_pitch"));
  m.def("get_cuda_device_for_current_OpenGL_context",
        &get_cuda_device_for_current_OpenGL_context,
        "Get CUDA device ID for the current OpenGL context");
  m.def("set_cuda_device_for_current_OpenGL_context",
        &set_cuda_device_for_current_OpenGL_context,
        "Set CUDA device for the current OpenGL context",
        pybind11::arg("device_id"));
}
