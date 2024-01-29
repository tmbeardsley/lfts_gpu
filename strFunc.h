// #####################################################################################
// Provides the public methods: void sample(...) and void save(...), 
// which take samples of the structure funtion, S(k), and save the spherically-averaged 
// S(k) to file
// #####################################################################################

#pragma once
#include <cuda.h>
#include <cufft.h>
#include "GPUerror.h"
#include <math.h>
#include <complex>
#include <fstream>
#include <iostream>




// Multiply and sum propagators for calculating either phiA[r] or phiB[r]
__global__ void add_norm(double *S_gpu, cufftDoubleComplex *wk_gpu, const int Mk)
{
	int const tid = threadIdx.x + blockIdx.x * blockDim.x;
	if (tid >= Mk) return;
	S_gpu[tid] += pow(wk_gpu[tid].x, 2.0) + pow(wk_gpu[tid].y, 2.0);
}



class strFunc {
    // strFunc-specific variables                  
    int TpB_;                       // GPU threads per block (default: 512)
    double *S_gpu_;                 // Collects sum of |wk_[k]| resulting from calls to: sample(double *w_gpu)
    double *K_;                     // Modulus of wavevector k
    double dK_;                     // Maximum allowed difference used to define like wave vectors for spherical averaging
    double coeff_;                  // A constant used in saving the structure function
    double chi_b_;                  // Bare chi of the simulation
    int Mk_;                        // Total number of field mesh points in k-space
    int *wt_;                       // Weighting of contribution from wavevector k
    int *P_;                        // Map transforming K_[] into ascending order
    int nsamples_;                  // Number of structure function samples taken
    cufftHandle wr_to_wk_;          // cufft plan transforming w-(r) to w-(k)
    std::complex<double> *wk_;      // Used to copy w-(k) from GPU to host
    cufftDoubleComplex* wk_gpu_;    // w-(k) on the GPU

    public:
        // Constructor
        strFunc(int *m, double *L, int M, int Mk, double CV, double chi_b, double dK = 1E-5, int TpB = 512) {
            TpB_ = TpB;
            Mk_ = Mk;
            dK_ = dK;
            chi_b_ = chi_b;
            nsamples_ = 0;
            coeff_ = CV/(chi_b*chi_b*M*M);
            wk_ = new std::complex<double>[Mk];
            K_ = new double[Mk];
            wt_ = new int[Mk];
            P_ = new int[Mk];

            // Allocate memory for w-(k) on the GPU
            GPU_ERR(cudaMalloc((void**)&wk_gpu_, Mk*sizeof(cufftDoubleComplex)));

            // Allocate memory for S(k) on the GPU
            GPU_ERR(cudaMalloc((void**)&S_gpu_, Mk*sizeof(double)));
            Array_init<<<(Mk_+TpB_-1)/TpB_, TpB_>>>(S_gpu_, 0.0, Mk);

            // Create a cufft plan for the Fourier transform on the GPU
            GPU_ERR(cufftPlan3d(&wr_to_wk_, m[0], m[1], m[2], CUFFT_D2Z));

            // Populate the wavevector modulus array, K_
            calcK(K_, m, L);

            // Populate the map, P_, which puts the wavevector moduli, K_, into ascending order
            calcSortedKMap(P_, K_);
        }

        // Sample norm(w-(k)) 
        void sample(double *w_gpu) {
            // Transform w-(r) to k-space to get w-(k)
            GPU_ERR(cufftExecD2Z(wr_to_wk_, w_gpu, wk_gpu_));

            // Sample the norm of w-(k) for each wavevector and add to its sum
            add_norm<<<(Mk_+TpB_-1)/TpB_, TpB_>>>(S_gpu_, wk_gpu_, Mk_);

            // Increment the number of samples
            nsamples_++;
        }

        // Output the spherically-averaged structure function to file
        void save(std::string fileName, int dp=8) {
            double S_sum = 0.0, *S;
            int k, n_same = 0;
            std::ofstream out_stream;

            out_stream.open(fileName);
            out_stream.precision(dp);
            out_stream.setf(std::ios::fixed, std::ios::floatfield);

            // Copy S_gpu to the host
            S = new double[Mk_];
            GPU_ERR(cudaMemcpy(S, S_gpu_, Mk_*sizeof(double), cudaMemcpyDeviceToHost));

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
        ~strFunc() {
            delete[] wk_;
            delete[] K_;
            delete[] wt_;
            delete[] P_;
            GPU_ERR(cufftDestroy(wr_to_wk_));
            GPU_ERR(cudaFree(wk_gpu_));
            GPU_ERR(cudaFree(S_gpu_));
        }




    private:
        // Calculate the wavevector moduli and store in K[]
        void calcK(double *K, int *_m, double *_L) {

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

        // Populate the map, P, to index K in ascending order
        void calcSortedKMap(int *P, double *K) {
            int k;

            // Initial (unsorted) index map of K-vectors
            for (k=0; k<Mk_; k++) P[k] = k;

            // Optimised bubble sort (simple since only called once)
            int n_new, n = Mk_;
            do {
                n_new = 0;
                for (k=1; k<n; k++) {
                    if (K[P[k-1]] > K[P[k]]) {
                        swap(P[k-1], P[k]);
                        n_new = k;
                    }
                }
                n = n_new;
            } while (n > 1);
        }

        // Swap two integers
        void swap(int &i, int &j) {
            int temp = i;
            i = j;
            j = temp;
        }

};
