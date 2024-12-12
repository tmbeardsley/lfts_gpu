// #######################################################################################
// Provides the public method: void fwd(...), which takes the propagators of the previous
// monomer as input and returns the propagators of the next monomer as output
// #######################################################################################

#include "step.h"


// Element by element multiplication of complex array, a[], by double array, b[]
__global__ void Mult_self(cufftDoubleComplex *a, const double *b, int const M)
{
	int const tid = threadIdx.x + blockIdx.x * blockDim.x;

	if (tid >= M) return;
	a[tid].x *= b[tid];
	a[tid].y *= b[tid];
}


// Element by element multiplication of double array, a[], by double array, b[]
__global__ void Mult_self(double *a, const double *b, int const M)
{
	int const tid = threadIdx.x + blockIdx.x * blockDim.x;

	if (tid >= M) return;
	a[tid] *= b[tid];
}


// Constructor
step::step(int NA, int NB, int *m, double *L, int M, int Mk, int TpB) :
    TpB_(TpB),
    g_gpu_(create_unique_cuda_memory<double>(2*Mk)),                // g_gpu_ contains two copies of g[] so that q1[k] and q2[k] can be multiplied on the GPU at the same time
    qk_gpu_(create_unique_cuda_memory<cufftDoubleComplex>(2*Mk)),   // Allocate memory for q1[k] and q2[k], stored in contigious memory
    qr_to_qk_(new cufftHandle, cufftDeleter()),
    qk_to_qr_(new cufftHandle, cufftDeleter()),
    NA_(NA),
    NB_(NB),
    M_(M),
    Mk_(Mk),
    m_(std::make_unique<int[]>(3))
{

    for (int i=0; i<3; i++) m_[i] = m[i];

    // Calculate the lookup table for g (copied to gpu in function for box move to be added later)
    update_g_lookup(L);

    // Configure cufft plans. cufftPlanMany used for batched processing
    GPU_ERR(cufftPlanMany(qr_to_qk_.get(), 3, m_.get(), NULL, 1, 0, NULL, 1, 0, CUFFT_D2Z, 2));
    GPU_ERR(cufftPlanMany(qk_to_qr_.get(), 3, m_.get(), NULL, 1, 0, NULL, 1, 0, CUFFT_Z2D, 2));
}


// Calculate propagators for the next monomer given propagators of previous monomer
// q_in  = q{i}(r), q^{N+1-i}(r)
// q_out = q{i+1}(r), q^{N-i}(r)
// h_gpu = hA_gpu, hB_gpu
void step::fwd(double* q_in, double* q_out, double *h_gpu, int i)
{
    // Fourier transform q{i}(r),q^{N+1-i}(r) to k-space: qk_gpu_(k)
    GPU_ERR(cufftExecD2Z(*(qr_to_qk_.get()), q_in, qk_gpu_.get()));

    // Multiply qk_gpu_(k) by g_gpu_(k)
    Mult_self <<<(2*Mk_+TpB_-1)/TpB_, TpB_>>> (qk_gpu_.get(), g_gpu_.get(), 2*Mk_);

    // Fourier transform qk_gpu_(k) to real-space: q_out[r]
    GPU_ERR(cufftExecZ2D(*(qk_to_qr_.get()), qk_gpu_.get(), q_out));

    // Multiply q_out[r] by hA[r] or hB[r] (depending on monomer index) to get q{i+1}(r)
    if (i < NA_) Mult_self<<<(M_+TpB_-1)/TpB_,TpB_>>>(q_out,h_gpu,M_);
    else Mult_self<<<(M_+TpB_-1)/TpB_,TpB_>>>(q_out,h_gpu+M_,M_);

    // Multiply q_out[r+M_] by hA[r] or hB[r] (depending on monomer index) to get q^{N-i}(r)
    if (i < NB_) Mult_self<<<(M_+TpB_-1)/TpB_,TpB_>>>(q_out+M_,h_gpu+M_,M_);
    else Mult_self<<<(M_+TpB_-1)/TpB_,TpB_>>>(q_out+M_,h_gpu,M_);
}


// Destructor
step::~step() {
    //GPU_ERR(cufftDestroy(qr_to_qk_));
    //GPU_ERR(cufftDestroy(qk_to_qr_));
}


// Calculate the Boltzmann weight of the bond potential in k-space, _g[k]
void step::update_g_lookup(double *L) {
    int K0, K1, k, N=NA_+NB_;
    double K, kx_sq, ky_sq, kz_sq;
    std::unique_ptr<double[]> g = std::make_unique<double[]>(Mk_);

    for (int k0=-(m_[0]-1)/2; k0<=m_[0]/2; k0++) {
        K0 = (k0<0)?(k0+m_[0]):k0;
        kx_sq = k0*k0/(L[0]*L[0]);

        for (int k1=-(m_[1]-1)/2; k1<=m_[1]/2; k1++) {
            K1 = (k1<0)?(k1+m_[1]):k1;
            ky_sq = k1*k1/(L[1]*L[1]);

            for (int k2=0; k2<=m_[2]/2; k2++) {
                kz_sq = k2*k2/(L[2]*L[2]);
                k = k2 + (m_[2]/2+1)*(K1+m_[1]*K0);
                K = 2*M_PI*pow(kx_sq+ky_sq+kz_sq,0.5); 
                g[k] = exp(-K*K/(6.0*N))/M_; 
            }
        }
    }
    GPU_ERR(cudaMemcpy(g_gpu_.get(), g.get(), Mk_*sizeof(double), cudaMemcpyHostToDevice));
    GPU_ERR(cudaMemcpy(g_gpu_.get()+Mk_, g.get(), Mk_*sizeof(double), cudaMemcpyHostToDevice));
}

