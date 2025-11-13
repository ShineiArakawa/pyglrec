/**
 * @file cuda_gl_interop.cpp
 * @author Shinei Arakawa (sarakawalab@gmail.com)
 * @brief CUDA-OpenGL inter-operability functions
 *
 * @copyright This file includes modified codes from:
 *   Erik Härkönen's 'PyViewer' library (licensed under CC BY-NC-SA 4.0, https://creativecommons.org/licenses/by-nc-sa/4.0/): https://github.com/harskish/pyviewer.git
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
#include <rgba_to_nv12.cuh>
#include <sstream>
#include <tuple>
#include <utils.hpp>

// --------------------------------------------------------------------------------------------------------------------------------
// Device memory management

// Linear 1D device memory allocation and deallocation

uint64_t allocate_device_memory_1d(size_t size_in_bytes) {
  void* ptr = nullptr;
  cudaErrors(cudaMalloc(&ptr, size_in_bytes));
  cudaErrors(cudaDeviceSynchronize());
  return reinterpret_cast<uint64_t>(ptr);
}

void free_device_memory(uint64_t d_ptr) {
  if (d_ptr == 0) {
    return;
  }
  cudaErrors(cudaFree(reinterpret_cast<void*>(d_ptr)));
  cudaErrors(cudaDeviceSynchronize());
}

// Linear 2D device memory allocation

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

// --------------------------------------------------------------------------------------------------------------------------------
// Texture copy with YUV conversion

void copy_texture_to_cuda_memory_with_YUV_conversion(
    uint64_t texture_id,
    int width,
    int height,
    uint64_t cuda_tex_ptr,
    uint64_t cuda_tex_pitch,
    uint64_t d_ptr) {
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

  // Copy and convert RGBA to NV12
  rgba_to_nv12(cuda_tex_ptr,
               cuda_tex_pitch,
               width,
               height,
               d_ptr);

  // Unmap the resource
  cudaErrors(cudaGraphicsUnmapResources(1, &cuda_resource, 0));
  cudaErrors(cudaGraphicsUnregisterResource(cuda_resource));

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
  m.def("copy_texture_to_cuda_memory_with_YUV_conversion",
        &copy_texture_to_cuda_memory_with_YUV_conversion,
        "Copy OpenGL texture to CUDA device memory with YUV conversion",
        pybind11::arg("texture_id"),
        pybind11::arg("width"),
        pybind11::arg("height"),
        pybind11::arg("cuda_tex_ptr"),
        pybind11::arg("cuda_tex_pitch"),
        pybind11::arg("d_ptr"));
}
