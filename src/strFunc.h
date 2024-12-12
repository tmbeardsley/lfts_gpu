// #####################################################################################
// Provides the public methods: void sample(...) and void save(...), 
// which take samples of the structure funtion, S(k), and save the spherically-averaged 
// S(k) to file.
// S(k) should only be calculated in simulations keeping L[] and XbN constant.
// #####################################################################################

#pragma once
#include <cuda.h>
#include <cufft.h>
#include "gpu_helpers/GPUerror.h"
#include "GPUkernels.h"
#include <math.h>
#include <complex>
#include <fstream>
#include <numeric>
#include <algorithm>
#include <memory>
#include "gpu_helpers/cuda_smart_pointer.h"
#include "gpu_helpers/cufftHelpers.h"

class strFunc {

    int TpB_;                                               // GPU threads per block (default: 512)
    unique_cuda_ptr<double> S_gpu_;                         // Collects sum of |wk_[k]| resulting from calls to: sample(double *w_gpu)
    std::unique_ptr<double[]> K_;                           // Modulus of wavevector k
    double dK_;                                             // Maximum allowed difference used to define like wave vectors for spherical averaging
    double coeff_;                                          // A constant used in saving the structure function
    std::unique_ptr<int[]> wt_;                             // Weighting of contribution from wavevector k           
    std::unique_ptr<int[]> P_;                              // Map transforming K_[] into ascending order
    int nsamples_;                                          // Number of structure function samples taken
    std::unique_ptr<std::complex<double>[]> wk_;            // Used to copy w-(k) from GPU to host
    unique_cuda_ptr<cufftDoubleComplex> wk_gpu_;            // w-(k) on the GPU
    std::unique_ptr<cufftHandle, cufftDeleter> wr_to_wk_;   // cufft plan transforming w-(r) to w-(k)

    // Simulation constants derived from the input file (see lfts_params.h for details)
    double chi_b_;
    int Mk_;

    public:
        // Constructor
        strFunc(int *m, double *L, int M, int Mk, double CV, double chi_b, int TpB = 512, double dK = 1E-5);

        // Sample norm(w-(k)) 
        void sample(double *w_gpu);

        // Output the spherically-averaged structure function to file
        void save(std::string fileName, int dp=8);

        // Destructor
        ~strFunc();


    private:
        // Calculate the wavevector moduli and store in K[]
        void calcK(double *K, int *_m, double *_L);

};
