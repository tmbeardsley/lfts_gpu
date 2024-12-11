// ######################################################
// Provides public method: calc_concs(double *w_gpu), 
// to calculate concentrations (used in Anderson mixing)
// ######################################################

#include "diblock.h"


// Calculate hA[r] and hB[r]: h -> hA[0], h+M -> hB[0] 
static __global__ void prepare_h(double *h, double *w_gpu, const int N, const int M)
{
	double *wm=w_gpu, *wp=w_gpu+M;
	int const tid = threadIdx.x + blockIdx.x * blockDim.x;

	if (tid >= 2*M) return;
	if (tid < M) h[tid] = exp(-(wm[tid]+wp[tid])/N);
	else h[tid] = exp(-(-wm[tid-M]+wp[tid-M])/N);
}


// Normalise concentrations phi-(r) and phi+(r): w_gpu+2*M -> phim[0],  w_gpu+3*M -> phip[0].
static __global__ void normalize_phi(double *w_gpu, double *h, const double Q, const double N, const int M)
{
    double *phiA=h, *phiB=h+M, *hA=h, *hB=h+M;
    double *phim=w_gpu+2*M, *phip=w_gpu+3*M;
	int const tid = threadIdx.x + blockIdx.x * blockDim.x;

	if (tid >= M) return;
	phiA[tid] = phim[tid]/(N*Q*hA[tid]);
	phiB[tid] = phip[tid]/(N*Q*hB[tid]);
	phim[tid] = phiA[tid] - phiB[tid];
	phip[tid] = phiA[tid] + phiB[tid];
}


// Multiply and sum propagators for calculating either phiA[r] or phiB[r]
static __global__ void sum_phi(double *phi_gpu, double *q1_gpu, double *q2_gpu, const int M)
{
	int const tid = threadIdx.x + blockIdx.x * blockDim.x;
	if (tid >= M) return;
	phi_gpu[tid] += q1_gpu[tid]*q2_gpu[tid];
}


// Constructor
diblock::diblock(int NA, int NB, int *m, double *L, int M, int Mk, int TpB) :
    TpB_(TpB),
    qr_gpu_(create_unique_cuda_memory<double>(2*(NA+NB+1)*M)),
    h_gpu_(create_unique_cuda_memory<double>(2*M)),
    q1_(std::make_unique<double*[]>(NA+NB+1)),
    q2_(std::make_unique<double*[]>(NA+NB+1)),
    Step_(std::make_unique<step>(NA, NB, m, L, M, Mk, TpB)),         // New step object containing methods to get next monomer's propagators
    M_(M),
    NA_(NA),
    N_(NA+NB)
{

    // Assign pointers such that q_{1}(r) and q_{N}(r) are in contigious memory,
    // as are q_{2}(r) and q_{N-1}(r), q_{3}(r) and q_{N-2}(r)... etc. (required for cufftPlanMany())
    for (int i=1; i<=N_; i++) {
        q1_[i] = qr_gpu_.get() + 2*i*M;
        q2_[N_+1-i] = qr_gpu_.get() + (2*i+1)*M;
    }

}


// Calculates phi-(r) and phi+(r): w+2*M -> phi-(0), w+3*M -> phi+(0).
// Returns ln(Q)
double diblock::calc_concs(double *w_gpu) {
    int i;
    double Q;                           // Single-chain partition function
    double *phiA_gpu=w_gpu+2*M_;
    double *phiB_gpu=w_gpu+3*M_;

    // Calculate hA[r] and hB[r] on the GPU
    prepare_h<<<(2*M_+TpB_-1)/TpB_, TpB_>>>(h_gpu_.get(), w_gpu, N_, M_);

    // Set initial conditions: q[1][r]=hA[r] and q^[N][r]=hB[r] for all r
    Array_copy<<<(2*M_+TpB_-1)/TpB_, TpB_>>>(q1_[1], h_gpu_.get(), 2*M_);

    // Step the propagators q1 and q2 for each subsequent monomer (note q[i],q^[N+1-i]... contigious in memory)
    for (i=1; i<N_; i++) Step_->fwd(q1_[i], q1_[i+1], h_gpu_.get(), i);

    // Calculate single-chain partition function using a Thrust reduction sum
    thrust::device_ptr<double> dp = thrust::device_pointer_cast(q1_[N_]);
    Q = thrust::reduce(dp, dp+M_, 0.0, thrust::plus<double>());
    Q /= M_;

    // Calculate concentrations using custom CUDA kernels
    Array_init<<<(2*M_+TpB_-1)/TpB_, TpB_>>>(phiA_gpu, 0.0, 2*M_);
    for (i=1; i<=NA_; i++) sum_phi<<<(M_+TpB_-1)/TpB_, TpB_>>>(phiA_gpu, q1_[i], q2_[i], M_);
    for (i=NA_+1; i<=N_; i++) sum_phi<<<(M_+TpB_-1)/TpB_, TpB_>>>(phiB_gpu, q1_[i], q2_[i], M_);
    normalize_phi<<<(M_+TpB_-1)/TpB_, TpB_>>>(w_gpu, h_gpu_.get(), Q, N_, M_);

    return log(Q);
}


// Destructor
diblock::~diblock() {

}
