#pragma once

#include <cuda.h>
#include <memory>
#include "GPUerror.h"


// Custom deleter for unique_ptr
template <typename T>
struct CudaDeleter {
    void operator()(T* ptr) const {
        if (ptr) GPU_ERR(cudaFree(ptr));
    }
};


// Template alias for unique_ptr with a deleter for CUDA memory
template <typename T>
using unique_cuda_ptr = std::unique_ptr<T[], CudaDeleter<T>>;


// Create unique_ptr with CUDA memory
template <typename T>
unique_cuda_ptr<T> create_unique_cuda_memory(size_t size) {
    T* raw_ptr;
    GPU_ERR(cudaMalloc((void**)&raw_ptr, size * sizeof(T)));
    return unique_cuda_ptr<T>(raw_ptr);
}
