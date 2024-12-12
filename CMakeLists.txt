cmake_minimum_required(VERSION 3.18)
project(fts_gpu LANGUAGES CXX CUDA)

# Set CUDA standard and specify the architecture flags
set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Specify the GPU architectures to compile for
set(CMAKE_CUDA_ARCHITECTURES all)

# Find required packages
find_package(CUDAToolkit REQUIRED)
find_package(GSL REQUIRED)

# Add the source file for the CUDA code
add_executable(lfts-gpu 
                ./src/fts_gpu.cu 
                ./src/diblock.cu 
                ./src/GPUkernels.cu 
                ./src/step.cu 
                ./src/anderson.cu 
                ./src/field_generator.cc 
                ./src/file_IO.cu
                ./src/langevin.cu
                ./src/strFunc.cu
                ./src/lfts_params.cc
                ./src/lfts_simulation.cu)

# Link required libraries (cuFFT, GSL, GSLcblas, cuRAND)
target_link_libraries(lfts-gpu PRIVATE
    CUDA::cufft       # CUDA FFT library
    GSL::gsl          # GSL (GNU Scientific Library)
    CUDA::curand      # CUDA random number generation library
)
