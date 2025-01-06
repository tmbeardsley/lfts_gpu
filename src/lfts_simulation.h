// ######################################################################################
// Exposes public methods to perform an L-FTS simulation: equilibrate() and statistics()
// ######################################################################################

#pragma once
#include <stdlib.h>
#include <string>
#include <cuda.h>
#include "gpu_helpers/GPUerror.h"
#include <iostream>
#include <fstream>
#include "diblock.h"
#include "anderson.h"
#include "strFunc.h"
#include <thrust/reduce.h>
#include <thrust/transform_reduce.h>
#include <curand.h>
#include <iomanip>
#include <limits>
#include "field_generator.h"
#include "langevin.h"
#include "lfts_params.h"
#include "file_IO.h"
#include <memory>
#include "gpu_helpers/cuda_smart_pointer.h"

#ifdef VIEW_FIELD
    // Include classes and libraries necessary for visualisation
    #include "viewFieldClass.h"
    #include <mutex>
    #include <thread>
#endif



class lfts_simulation {
    unique_cuda_ptr<double> w_gpu_;         // GPU array containing: N*w-(r), N*w+(r), phi-(r), phi+(r)
    std::unique_ptr<diblock> dbc_;          // Diblock object for calculating phi-(r) and phi+(r)
    std::unique_ptr<anderson> AM_;          // Anderson mixing object to solve for w+(r)
    std::unique_ptr<langevin> Langevin_;    // Langevin object to update w-(r) at each step
    std::unique_ptr<strFunc> Sk_;           // StrFunc object for dealing with sampling and calculating the structure function
    curandGenerator_t RNG_;                 // Random number generator for the GPU

    #ifdef VIEW_FIELD
        viewFieldClass* field_viewer_ = nullptr;
    #endif


    int M_;                     // Total number of field mesh points (constant - contained in lfts_params object but copied for tidier code)

    public:
        lfts_simulation(std::string inputFile, int TpB=512);

        // Destructor
        ~lfts_simulation();

        // Equilibration loop, during which statistics are NOT sampled
        void equilibrate();

        // Statistics loop, during which statistics are sampled
        void statistics();

        // Calculate the diblock copolymer Hamiltonian
        double getH();

        std::unique_ptr<lfts_params> P_;        // Object to hold the simulation parameters - automatically updates derived parameters

        #ifdef VIEW_FIELD
            // Supply data to the field viewer for visualisation
            void update_field_viewer_data(viewFieldClass* field_viewer, double* w_gpu, int M);
            void set_field_viewer(viewFieldClass* field_viewer);
        #endif


    private:
        // Save data in a standard format to be used as in input file
        void saveStdOutputFile(std::string fileName);

        // Generate an initial w-(r) field (sets w+(r) = 0)
        void generate_field(double *w, int loadType);

};