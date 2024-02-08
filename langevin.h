// #############################################################################
// Performs a langevin update of w-(r) and keeps track of symmetrised noise
// #############################################################################
#pragma once

#include <cuda.h>
#include "GPUerror.h"
#include <curand.h>
#include "anderson.h"
#include "diblock.h"
#include "lfts_params.h"

// Langevin update of w-(r) on the GPU using symmetrised noise
__global__ void langevin_sym(double *w_gpu, double *noise_gpu_new, double *noise_gpu_prev, const double XbN, const double dt, const int M)
{
	int const tid = threadIdx.x + blockIdx.x * blockDim.x;
	if (tid >= M) return;
	w_gpu[tid] += -(w_gpu[tid+2*M]+2*w_gpu[tid]/XbN)*dt+0.5*(noise_gpu_prev[tid]+noise_gpu_new[tid]);
}




class langevin {
    int    TpB_;                // GPU threads per block (default: 512)
    double *noise_gpu_;         // Array holding random noise for current step and previous step
    double *noise_gpu_new_;     // Pointer to portion of memory for new noise in noise_gpu_[]
    double *noise_gpu_prev_;    // Pointer to portion of memory for previous noise in noise_gpu_[]

    // Simulation constants derived from the input file (see lfts_params.h for details)
    int    M_;

    public:
        langevin(curandGenerator_t &RNG, double sigma, int M, int TpB=512) {
            TpB_ = TpB;
            M_ = M;

            // Allocate memory for Gaussian random noise on the GPU
            GPU_ERR(cudaMalloc((void**)&noise_gpu_,2*M_*sizeof(double)));

            // Generate initial "previous" Gaussian random noise on the gpu
            curandGenerateNormalDouble(RNG, noise_gpu_, M_, 0.0, sigma);
            noise_gpu_prev_ = noise_gpu_;
            noise_gpu_new_ = noise_gpu_ + M_;
        }

        ~langevin() {
            GPU_ERR(cudaFree(noise_gpu_));
        }

        // Perform a Langevin update of the fields using symmetrised noise
        void step_wm(double* w_gpu, curandGenerator_t &RNG, double XbN, double sigma, double dt)
        {
            double *ptr_tmp;

            // Create new random noise on the GPU for the call to langevin()
            curandGenerateNormalDouble(RNG, noise_gpu_new_, M_, 0.0, sigma);

            // Perform the Langevin step on the GPU
            langevin_sym<<<(M_+TpB_-1)/TpB_,TpB_>>>(w_gpu, noise_gpu_new_, noise_gpu_prev_, XbN, dt, M_);

            // Swap noise pointer positions to avoid shifting (copying) data every step
            ptr_tmp = noise_gpu_prev_;
            noise_gpu_prev_ = noise_gpu_new_;
            noise_gpu_new_ = ptr_tmp;
        }

};