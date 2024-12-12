// #####################################################################
// Functions to manage problems arising from CUDA and the cuFFT library
// #####################################################################

#pragma once
#include <cuda.h>
#include <cufft.h>
#include <stdexcept>
#include <iostream>
#include <string>


inline void HandleError(cudaError_t err, const char* file, const int line)
{
    if (err != cudaSuccess) {
        std::cerr << cudaGetErrorString(err) << " in " << file << " at line " << line << std::endl;
        throw std::runtime_error(cudaGetErrorString(err));
    }
}


inline void HandleError(cufftResult err, const char* file, const int line)
{
    if (err != CUFFT_SUCCESS) {
        std::string error_message;
        switch(err) {
            case CUFFT_INVALID_PLAN:
                error_message = "CUFFT_INVALID_PLAN";
                break;
            case CUFFT_INVALID_VALUE:
                error_message = "CUFFT_INVALID_VALUE";
                break;
            case CUFFT_INTERNAL_ERROR:
                error_message = "CUFFT_INTERNAL_ERROR";
                break;
            case CUFFT_EXEC_FAILED:
                error_message = "CUFFT_EXEC_FAILED";
                break;
            case CUFFT_SETUP_FAILED:
                error_message = "CUFFT_SETUP_FAILED";
                break;
            default:
                error_message = "Unknown cuFFT error";
        }
        std::cerr << error_message << " in " << file << " at line " << line << std::endl;
        throw std::runtime_error(error_message);
    }
}

#define GPU_ERR( err ) (HandleError( err, __FILE__, __LINE__ ))