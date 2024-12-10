// ######################################################
// Provides public method: calc_concs(double *w_gpu), 
// to calculate concentrations (used in Anderson mixing)
// ######################################################

#pragma once
#include <cuda.h>
#include "GPUkernels.h"
#include "GPUerror.h"
#include "step.h"
#include <thrust/reduce.h>
#include <thrust/device_ptr.h>
#include <memory>


class diblock {

    // Diblock-specific variables
    int TpB_;
    double *qr_gpu_;                    // Pointer to GPU memory for propagators: q_{i}(r) and q^_{N+1-i}(r) are contigious in memory
    double *h_gpu_;                     // Pointer to GPU memory for hA(r) and hB(r)
    std::unique_ptr<double*[]> q1_;     // Array of pointers to q_{j=i}(r), where j is the monomer index and i is array index
    std::unique_ptr<double*[]> q2_;     // Array of pointers to q^_{j=N+1-i}(r), where j is the monomer index and i is array index
    std::unique_ptr<step> Step_;        // Step object to get propagators for the next monomer

    // Simulation constants derived from the input file (see lfts_params.h for details)
    int M_;
    int NA_;
    int N_;


    public:
        // Constructor
        diblock(int NA, int NB, int *m, double *L, int M, int Mk, int TpB=512);

        // Calculates phi-(r) and phi+(r): w+2*M -> phi-(0), w+3*M -> phi+(0).
        // Returns ln(Q)
        double calc_concs(double *w_gpu);

        // Destructor
        ~diblock();
};