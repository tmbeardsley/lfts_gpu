// #############################################################################
// Performs a langevin update of w-(r) and keeps track of symmetrised noise
// #############################################################################

#include "langevin.h"

// Langevin update of w-(r) on the GPU using symmetrised noise
static __global__ void langevin_sym(double *w_gpu, double *noise_gpu_new, double *noise_gpu_prev, const double XbN, const double dt, const int M)
{
	int const tid = threadIdx.x + blockIdx.x * blockDim.x;
	if (tid >= M) return;
	w_gpu[tid] += -(w_gpu[tid+2*M]+2*w_gpu[tid]/XbN)*dt+0.5*(noise_gpu_prev[tid]+noise_gpu_new[tid]);
}


langevin::langevin(curandGenerator_t &RNG, double sigma, int M, int TpB) :
    TpB_(TpB),
    noise_gpu_(create_unique_cuda_memory<double>(2*M)),
    M_(M)
{

    // Generate initial "previous" Gaussian random noise on the gpu
    curandGenerateNormalDouble(RNG, noise_gpu_.get(), M_, 0.0, sigma);
    noise_gpu_prev_ = noise_gpu_.get();
    noise_gpu_new_ = noise_gpu_.get() + M_;
}


langevin::~langevin() {

}


// Perform a Langevin update of the fields using symmetrised noise
void langevin::step_wm(double* w_gpu, curandGenerator_t &RNG, double XbN, double sigma, double dt)
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

