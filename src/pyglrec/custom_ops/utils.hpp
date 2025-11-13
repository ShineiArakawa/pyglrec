#pragma once

#include <cuda_runtime.h>

#include <iostream>

// --------------------------------------------------------------------------------------------------------------------------------
// Utilities

#define cudaErrors(...)                                                                  \
  do {                                                                                   \
    cudaError_t error = __VA_ARGS__;                                                     \
    if (error) {                                                                         \
      std::cout << cudaGetErrorName(error) << ": " << cudaGetErrorString(error) << "\n"; \
      std::cout << "while running " #__VA_ARGS__ "\n";                                   \
    }                                                                                    \
  } while (false)
