// #############################################################################
// Provides useful functions for file reading and writing.
// Not yet templated as current code version only has need to output doubles.
// #############################################################################
#pragma once
#include <fstream>
#include <iomanip>
#include <limits>
#include <cuda.h>
#include "gpu_helpers/GPUerror.h"

namespace file_IO {

    // Check whether a file exists
    bool isValidFile(std::string fileName);

    // Read an array from file
    void readArray(double *arr, std::string fileName, int n, int nIgnore=0);

    // Save a host array to file
    void saveArray(double *arr, std::string fileName, int n, bool append=false);

    // Save a GPU array to file
    void saveGPUArray(double *arr_gpu, std::string fileName, int n, bool append=false);

}