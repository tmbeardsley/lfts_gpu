// #############################################################################
// Performs a langevin update of w-(r) and keeps track of symmetrised noise
// #############################################################################
#pragma once

#include <cuda.h>
#include <curand.h>
#include "cuda_smart_pointer.h"



class langevin {
    int    TpB_;                            // GPU threads per block (default: 512)
    unique_cuda_ptr<double> noise_gpu_;     // Array holding random noise for current step and previous step
    double *noise_gpu_new_;                 // Pointer to portion of memory for new noise in noise_gpu_[]
    double *noise_gpu_prev_;                // Pointer to portion of memory for previous noise in noise_gpu_[]

    // Simulation constants derived from the input file (see lfts_params.h for details)
    int    M_;

    public:
        langevin(curandGenerator_t &RNG, double sigma, int M, int TpB=512);

        ~langevin();

        // Perform a Langevin update of the fields using symmetrised noise
        void step_wm(double* w_gpu, curandGenerator_t &RNG, double XbN, double sigma, double dt);
        
};