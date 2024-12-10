// ##########################
// Commonly used GPU kernels
// ##########################

#pragma once
#include <cuda.h>

// Set all elements of array, a, to the constant b
extern __global__ void Array_init(double *a, const double b, int const M);

// Set array, a, equal to array b. Add an optional constant, C, to each array element
extern __global__ void Array_copy(double *a, const double *b, int const M, double C = 0.0);
