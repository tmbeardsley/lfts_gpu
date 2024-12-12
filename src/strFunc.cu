// #####################################################################################
// Provides the public methods: void sample(...) and void save(...), 
// which take samples of the structure funtion, S(k), and save the spherically-averaged 
// S(k) to file.
// S(k) should only be calculated in simulations keeping L[] and XbN constant.
// #####################################################################################

#include "strFunc.h"


// Multiply and sum propagators for calculating either phiA[r] or phiB[r]
static __global__ void add_norm(double *S_gpu, cufftDoubleComplex *wk_gpu, const int Mk)
{
	int const tid = threadIdx.x + blockIdx.x * blockDim.x;
	if (tid >= Mk) return;
	S_gpu[tid] += pow(wk_gpu[tid].x, 2.0) + pow(wk_gpu[tid].y, 2.0);
}


// Constructor
strFunc::strFunc(int *m, double *L, int M, int Mk, double CV, double chi_b, int TpB, double dK) :
    TpB_(TpB),
    S_gpu_(create_unique_cuda_memory<double>(Mk)),
    K_(std::make_unique<double[]>(Mk)),
    dK_(dK),
    coeff_(CV/(chi_b*chi_b*M*M)),
    wt_(std::make_unique<int[]>(Mk)),
    P_(std::make_unique<int[]>(Mk)),
    nsamples_(0),
    wk_(std::make_unique<std::complex<double>[]>(Mk)),
    wk_gpu_(create_unique_cuda_memory<cufftDoubleComplex>(Mk)),
    chi_b_(chi_b),
    Mk_(Mk),
    wr_to_wk_(new cufftHandle, cufftDeleter())
    //wr_to_wk_(std::make_unique<>())
{

    // Allocate memory for S(k) on the GPU
    Array_init<<<(Mk_+TpB_-1)/TpB_, TpB_>>>(S_gpu_.get(), 0.0, Mk);

    // Create a cufft plan for the Fourier transform on the GPU
    GPU_ERR(cufftPlan3d(wr_to_wk_.get(), m[0], m[1], m[2], CUFFT_D2Z));

    // Populate the wavevector modulus array, K_
    calcK(K_.get(), m, L);

    // Populate the map, P_, which puts the wavevector moduli, K_, into ascending order
    std::iota(P_.get(), P_.get() + Mk_, 0);
    std::stable_sort(P_.get(), P_.get()+Mk_, [this](size_t i, size_t j) {return K_[i] < K_[j];});
}


// Sample norm(w-(k)) 
void strFunc::sample(double *w_gpu) {
    // Transform w-(r) to k-space to get w-(k)
    GPU_ERR(cufftExecD2Z(*(wr_to_wk_.get()), w_gpu, wk_gpu_.get()));

    // Sample the norm of w-(k) for each wavevector and add to its sum
    add_norm<<<(Mk_+TpB_-1)/TpB_, TpB_>>>(S_gpu_.get(), wk_gpu_.get(), Mk_);

    // Increment the number of samples
    nsamples_++;
}


// Output the spherically-averaged structure function to file
void strFunc::save(std::string fileName, int dp) {
    double S_sum = 0.0, *S;
    int k, n_same = 0;
    std::ofstream out_stream;

    out_stream.open(fileName);
    out_stream.precision(dp);
    out_stream.setf(std::ios::fixed, std::ios::floatfield);

    // Copy S_gpu to the host
    S = new double[Mk_];
    GPU_ERR(cudaMemcpy(S, S_gpu_.get(), Mk_*sizeof(double), cudaMemcpyDeviceToHost));

    // Spherical average of S(k)
    for (k=0; k<Mk_; k++) {
        // Take into account vector weighting from the FFT and sum S for repeated K-vectors
        S_sum += wt_[P_[k]] * ((coeff_/nsamples_)*S[P_[k]] - 0.5/chi_b_);
        n_same += wt_[P_[k]];

        // Output value for current K-vector when difference in K exceeds tolerence dK_
        if ( (k==Mk_-1) || (fabs(K_[P_[k+1]]-K_[P_[k]]) > dK_) ) {
            out_stream << K_[P_[k]] << "\t" << S_sum/n_same << std::endl;

            // Reset summations for next K-vector
            S_sum = 0.0;
            n_same = 0;
        }
    } 
    out_stream.close();

    delete[] S;
}


// Destructor
strFunc::~strFunc() {
    //GPU_ERR(cufftDestroy(wr_to_wk_));
}


// Calculate the wavevector moduli and store in K[]
void strFunc::calcK(double *K, int *_m, double *_L) {

    int K0, K1, k;
    double kx_sq, ky_sq, kz_sq;

    for (k=0; k<Mk_; k++) wt_[k]=2;

    for (int k0=-(_m[0]-1)/2; k0<=_m[0]/2; k0++) {
        K0 = (k0<0)?(k0+_m[0]):k0;
        kx_sq = k0*k0/(_L[0]*_L[0]);

        for (int k1=-(_m[1]-1)/2; k1<=_m[1]/2; k1++) {
            K1 = (k1<0)?(k1+_m[1]):k1;
            ky_sq = k1*k1/(_L[1]*_L[1]);

            for (int k2=0; k2<=_m[2]/2; k2++) {
                kz_sq = k2*k2/(_L[2]*_L[2]);
                k = k2 + (_m[2]/2+1)*(K1+_m[1]*K0);
                K[k] = 2*M_PI*pow(kx_sq+ky_sq+kz_sq,0.5); 
                if ((k2==0)||(k2==_m[2]/2)) wt_[k]=1;
            }
        }
    }
}


