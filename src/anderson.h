// #######################################################
// Performs Anderson Mixing to solve for the W+(r) field
// #######################################################

#pragma once
#include "GPUerror.h"   // GPU error handling kernels
#include "GPUkernels.h" // GPU kernels
#include "diblock.h"
#include <thrust/transform_reduce.h>
#include <thrust/functional.h>
#include <thrust/device_vector.h>
#include <thrust/iterator/zip_iterator.h>
#include <gsl/gsl_linalg.h>
#include <gsl/gsl_blas.h>
#include <memory>
#include "cuda_smart_pointer.h"


class anderson {
    int TpB_;                   // GPU threads per block (default: 512)
    int nhMax_;                 // Maximum # histories (default: 10)

    // Mathematical arrays
    std::unique_ptr<std::unique_ptr<double[]>[]> DD_;
    std::unique_ptr<double[]> U_;
    std::unique_ptr<double[]> V_;
    std::unique_ptr<double[]> C_;
    unique_cuda_ptr<double> Dh_gpu_mem_;
    std::unique_ptr<double*[]> Dh_gpu_;
    unique_cuda_ptr<double> wh_gpu_mem_;
    std::unique_ptr<double*[]> wh_gpu_;
    std::unique_ptr<double[]> A_cpy_;
    std::unique_ptr<double[]> Y_cpy_;

    // Simulation constants derived from the input file (see lfts_params.h for details)
    int M_;


    public:
        // Constructor
        anderson(int M, int maxHist=10, int TpB=512);
        
        int mix(diblock *dbc, int maxIter, double errTol, double *w_gpu);

        // Destructor
        ~anderson();

        private:
            // Return a 2d array of dimensions (m,n)
            std::unique_ptr<std::unique_ptr<double[]>[]> array2d(const int m, const int n);

            void LUdecomp(double *A, double *Y, double *X, const int n);

};