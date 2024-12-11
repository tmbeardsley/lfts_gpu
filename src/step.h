// #######################################################################################
// Provides the public method: void fwd(...), which takes the propagators of the previous
// monomer as input and returns the propagators of the next monomer as output
// #######################################################################################

#pragma once
#include <cuda.h>
#include <cufft.h>
#include "GPUerror.h"
#include <cmath>
#include <memory>
#include "cuda_smart_pointer.h"


class step {
    // Step-specific variables
    int TpB_;                                       // GPU threads per block (default: 512)
    unique_cuda_ptr<double> g_gpu_;                 // Bond potential Boltzmann weight, Fourier transformed and /M_ on the GPU
    unique_cuda_ptr<cufftDoubleComplex> qk_gpu_;    // Fourier transforms of q1 and q2 on the GPU (for cufftPlanMany())
    cufftHandle qr_to_qk_;                          // cufft plan to transform q1[r] and q2[r] to k-space
    cufftHandle qk_to_qr_;                          // cufft plan to transform q1[k] and q2[k] to real-space

    // Simulation constants derived from the input file (see lfts_params.h for details)
    int NA_;
    int NB_;
    int M_;
    int Mk_;
    std::unique_ptr<int[]> m_;


    public:
        // Constructor
        step(int NA, int NB, int *m, double *L, int M, int Mk, int TpB=512);

        // Calculate propagators for the next monomer given propagators of previous monomer
        // q_in  = q{i}(r), q^{N+1-i}(r)
        // q_out = q{i+1}(r), q^{N-i}(r)
        // h_gpu = hA_gpu, hB_gpu
        void fwd(double* q_in, double* q_out, double *h_gpu, int i);

        // Destructor
        ~step();


    private:
        // Calculate the Boltzmann weight of the bond potential in k-space, _g[k]
        void update_g_lookup(double *L);

};




