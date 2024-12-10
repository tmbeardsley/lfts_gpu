// #######################################################
// Performs Anderson Mixing to solve for the W+(r) field
// #######################################################

#include "anderson.h"


// Typedefs for using Cuda Thrust library
typedef thrust::device_ptr<double> dp;
typedef thrust::tuple<dp,dp> t_dpD_dpD;
typedef thrust::zip_iterator<t_dpD_dpD> z_dpD_dpD;


// Perform simple mixing on the GPU
__global__ void simpleMix(double *wp_gpu, double *Dh_gpu_0, const double lambda, const int M)
{
	int const tid = threadIdx.x + blockIdx.x * blockDim.x;
	if (tid >= M) return;
	wp_gpu[tid] += lambda * Dh_gpu_0[tid];
}

// Perform Anderson mixing on the GPU
__global__ void AMmix(double *wp_gpu, double *wh_gpu_0, double *wh_gpu_n, double *Dh_gpu_0, double *Dh_gpu_n, const double Cn, const double lambda, const int M)
{
	int const tid = threadIdx.x + blockIdx.x * blockDim.x;
	if (tid >= M) return;
	wp_gpu[tid] += Cn * ( (wh_gpu_n[tid]+lambda*Dh_gpu_n[tid]) - (wh_gpu_0[tid]+lambda*Dh_gpu_0[tid]) );
}

// Multiply two doubles for thrust::transform_reduce()
struct mult_2dbls
{
	__device__
	double operator()(thrust::tuple<double, double> t) {
		return thrust::get<0>(t) * thrust::get<1>(t);
	}
};




// Constructor
anderson::anderson(int M, int maxHist, int TpB) :
    TpB_(TpB),
    nhMax_(maxHist),
    DD_(array2d(maxHist+1, maxHist+1)),
    U_(std::make_unique<double[]>(maxHist*maxHist)),
    V_(std::make_unique<double[]>(maxHist)),
    C_(std::make_unique<double[]>(maxHist)),
    Dh_gpu_(std::make_unique<double*[]>(maxHist+1)),
    wh_gpu_(std::make_unique<double*[]>(maxHist+1)),
    A_cpy_(std::make_unique<double[]>(maxHist*maxHist)),
    Y_cpy_(std::make_unique<double[]>(maxHist)),
    M_(M)
{
    // Mathematical array memory on the GPU
    GPU_ERR(cudaMalloc((void**)&Dh_gpu_mem_, (nhMax_+1)*M_*sizeof(double)));
    for (int i=0; i<nhMax_+1; i++) Dh_gpu_[i] = Dh_gpu_mem_ + i*M_;

    // Mathematical array memory on the GPU
    GPU_ERR(cudaMalloc((void**)&wh_gpu_mem_, (nhMax_+1)*M_*sizeof(double)));
    for (int i=0; i<nhMax_+1; i++) wh_gpu_[i] = wh_gpu_mem_ + i*M_;
}


int anderson::mix(diblock *dbc, int maxIter, double errTol, double *w_gpu) {
    double lambda, err=1.0, S1;
    int    k, m, n, nh;
    dp tPtr, tPtr2;
    t_dpD_dpD t;
    z_dpD_dpD z;

    for (k=1; k<maxIter && err>errTol; k++) {

        // Calculate concentrations. In: w-(r) and w+(r). Out: phi-(r) and phi+(r).
        dbc->calc_concs(w_gpu);

        // Copy arrays on the GPU
        Array_copy<<<(M_+TpB_-1)/TpB_, TpB_>>>(Dh_gpu_[0], w_gpu+3*M_, M_, -1.0);
        Array_copy<<<(M_+TpB_-1)/TpB_, TpB_>>>(wh_gpu_[0], w_gpu+M_, M_);

        // Sum of (phi-(r)-1.0)^2 performed on the GPU
        tPtr = thrust::device_pointer_cast(Dh_gpu_[0]);
        S1 = thrust::transform_reduce(tPtr, tPtr + M_, thrust::square<double>(), 0.0, thrust::plus<double>());

        // Update mixing error
        err = pow(S1/M_,0.5);
        lambda = 1.0-pow(0.9,double(k));

        // Update the number of histories
        nh = (k<nhMax_+1)?k-1:nhMax_;

        // Summations performed on the GPU for each history
        for (m=0; m<=nh; m++) {
            tPtr2 = thrust::device_pointer_cast(Dh_gpu_[m]);
            t = thrust::make_tuple(tPtr, tPtr2);
            z = thrust::make_zip_iterator(t);
            DD_[0][m] = thrust::transform_reduce(z, z+M_, mult_2dbls(), 0.0, thrust::plus<double>());
        }

        if (k<2) {
            // Simple mixing on the GPU
            simpleMix<<<(M_+TpB_-1)/TpB_, TpB_>>>(w_gpu+M_, Dh_gpu_[0], lambda, M_);
        } else {   
            // Anderson mixing
            for (m=1; m<=nh; m++) {
                V_[m-1] = DD_[0][0]-DD_[0][m];
                for (n=1; n<=m; n++) {
                    U_[(m-1)*nh+n-1] = U_[(n-1)*nh+m-1] = DD_[0][0]-DD_[0][m]-DD_[0][n]+DD_[n][m];
                }
            }

            // Solve for small matrix C_[] on the host using LU decomposition (U_[] and V_[] unchanged)
            LUdecomp(U_.get(),V_.get(),C_.get(),nh);

            // Initial simple mixing step: updates w+(r)
            simpleMix<<<(M_+TpB_-1)/TpB_, TpB_>>>(w_gpu+M_, Dh_gpu_[0], lambda, M_);

            // Perform Anderson Mixing for each history: updates w+(r)
            for (n=1; n<=nh; n++) {
                AMmix<<<(M_+TpB_-1)/TpB_, TpB_>>>(w_gpu+M_, wh_gpu_[0], wh_gpu_[n], Dh_gpu_[0], Dh_gpu_[n], C_[n-1], lambda, M_);
            }
        }

        // Field and deviation of current step become history n
        n=1+(k-1)%(nhMax_);
        Array_copy<<<(M_+TpB_-1)/TpB_, TpB_>>>(Dh_gpu_[n], Dh_gpu_[0], M_);
        Array_copy<<<(M_+TpB_-1)/TpB_, TpB_>>>(wh_gpu_[n], wh_gpu_[0], M_);
        DD_[n][n] = DD_[0][0];
        for (m=1; m<n; m++) DD_[m][n] = DD_[0][m];
        for (m=n+1; m<=nh; m++) DD_[n][m] = DD_[0][m];
    }

    return k;
}


// Destructor
anderson::~anderson() {
    GPU_ERR(cudaFree(Dh_gpu_mem_));
    GPU_ERR(cudaFree(wh_gpu_mem_));
}


// Function to create a 2D array using unique_ptr
std::unique_ptr<std::unique_ptr<double[]>[]> anderson::array2d(const int m, const int n) {
    auto a = std::make_unique<std::unique_ptr<double[]>[]>(m);  // Array of unique_ptr
    for (int i = 0; i < m; ++i) {
        a[i] = std::make_unique<double[]>(n);  // Allocate array for each row
    }
    return a;
}


void anderson::LUdecomp(double *A, double *Y, double *X, const int n) {
    int s;

    // Make copies of A and Y since LU will changes matrix/vector contents.
    for (int i=0; i<n; i++) {
        Y_cpy_[i] = Y[i];
        for (int j=0; j<n; j++) {
            A_cpy_[i*n+j] = U_[i*n+j];
        }
    }

    // Create matrix and vector views to use gsl library functions
    gsl_matrix_view a = gsl_matrix_view_array(A_cpy_.get(), n, n);
    gsl_vector_view y = gsl_vector_view_array(Y_cpy_.get(), n);
    gsl_vector_view x = gsl_vector_view_array(X, n);
    gsl_permutation *p = gsl_permutation_alloc(n);

    // Solve for x using LU decomposition via gsl library
    gsl_linalg_LU_decomp(&a.matrix, p, &s);
    gsl_linalg_LU_solve(&a.matrix, p, &y.vector, &x.vector);
    gsl_permutation_free(p);
}

