// #####################################################################
// Functions to manage problems arising from CUDA and the cuFFT library
// #####################################################################

#pragma once
#include <cuda.h>
#include <cufft.h>
#include <stdio.h>

#define GPU_ERR( err ) (HandleError( err, __FILE__, __LINE__ ))

extern void HandleError(cudaError_t err, char const * const file, int const line);

extern void HandleError(cufftResult err, char const * const file, int const line);



