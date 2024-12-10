// ######################################################################################
// Exposes public methods to perform an L-FTS simulation: equilibrate() and statistics()
// ######################################################################################

#include "lfts_simulation.h"



lfts_simulation::lfts_simulation(std::string inputFile, int TpB) 
{

    // Check that input file exists before proceeding
    if (!file_IO::isValidFile(inputFile)) {
        std::cout << "ERROR => Cannot open the L-FTS input file." << std::endl;
        exit(1);
    }

    // Read simulation parameters from the input file and allocate temporary host memory for fields
    P_ = std::make_unique<lfts_params>(inputFile);
    P_->outputParameters();
    M_=P_->M();
    std::unique_ptr<double[]> w = std::make_unique<double[]>(2*M_);

    // Set up random number generator
    curandCreateGenerator(&RNG_, CURAND_RNG_PSEUDO_DEFAULT);
    int iseed = time(NULL);
    iseed = 123456789;
    curandSetPseudoRandomGeneratorSeed(RNG_, iseed);
    std::cout << "RNG seed: " << iseed << std::endl;

    // Allocate memory for field array on the GPU
    GPU_ERR(cudaMalloc((void**)&w_gpu_, 4*M_*sizeof(double)));

    // Create a new diblock object
    std::cout << "creating diblock object..." << std::endl;
    dbc_ = std::make_unique<diblock>(P_->NA(), P_->NB(), P_->m(), P_->L(), M_, P_->Mk(), TpB);

    // Create a new anderson mixing class
    std::cout << "creating anderson object..." << std::endl;
    AM_ = std::make_unique<anderson>(M_, 10, TpB);

    // Set up a langevin object to upstate w-(r) at each step
    std::cout << "creating langevin object..." << std::endl;
    Langevin_ = std::make_unique<langevin>(RNG_, P_->sigma(), M_, TpB);

    // Create new structure strFunc object
    std::cout << "creating strFunc object..." << std::endl;
    //Sk_ = new strFunc(P_->m(), P_->L(), M_, P_->Mk(), P_->n(), P_->XbN(), TpB);
    Sk_ = std::make_unique<strFunc>(P_->m(), P_->L(), M_, P_->Mk(), P_->n(), P_->XbN(), TpB);

    // Read w-[r] and w+[r] from the input file
    if (P_->loadType() == 1) { 
        std::cout << "generating input field..." << std::endl;
        file_IO::readArray(w.get(), inputFile, 2*M_, 3);
    }
    else generate_field(w.get(), P_->loadType());

    // Copy w-(r) and w+(r) from host to GPU
    GPU_ERR(cudaMemcpy(w_gpu_, w.get(), 2*M_*sizeof(double), cudaMemcpyHostToDevice));

    // Perform an initial mix to get phi-(r) and phi+(r) from the input fields
    std::cout << "Initial Anderson mix..." << std::endl;
    AM_ -> mix(dbc_.get(), 200, 1e-4, w_gpu_);

    // Output initial fields
    saveStdOutputFile("w_0");
    file_IO::saveGPUArray(w_gpu_+2*M_, "phi_0", 2*M_);
}


// Destructor
lfts_simulation::~lfts_simulation() {
    GPU_ERR(cudaFree(w_gpu_));
}


// Equilibration loop, during which statistics are NOT sampled
void lfts_simulation::equilibrate() {
    int it;
    for (it=1; it<=P_->equil_its(); it++) {

        // Perform a Langevin step with symmetrised noise to update w-(r)
        Langevin_->step_wm(w_gpu_, RNG_, P_->XbN(), P_->sigma(), P_->dt());

        // Calculate saddle point value of w+(r), phi-(r) and phi+(r)
        AM_->mix(dbc_.get(), 200, 1e-4, w_gpu_);
        std::cout << "lnQ = " << dbc_->calc_concs(w_gpu_) << std::endl;

        // Save to file every save_freq_ steps
        if (it%P_->save_freq()==0) { 
            saveStdOutputFile("w_eq_" + std::to_string(it));
            file_IO::saveGPUArray(w_gpu_+2*M_, "phi_eq_"+std::to_string(it), 2*M_);
        }
    }
    // Final save to file at end of equilibration period
    saveStdOutputFile("w_eq_" + std::to_string(it-1));
    file_IO::saveGPUArray(w_gpu_+2*M_, "phi_eq_"+std::to_string(it-1), 2*M_);
}


// Statistics loop, during which statistics are sampled
void lfts_simulation::statistics() {
    int it;
    for (it=1; it<=P_->sim_its(); it++) {

        // Perform a Langevin step with symmetrised noise to update w-(r)
        Langevin_->step_wm(w_gpu_, RNG_, P_->XbN(), P_->sigma(), P_->dt());

        // Calculate saddle point value of w+(r), phi-(r) and phi+(r)
        AM_->mix(dbc_.get(), 200, 1e-4, w_gpu_);
        std::cout << "lnQ = " << dbc_->calc_concs(w_gpu_) << std::endl;

        // Sample statistics every sample_freq_ steps
        if (it%P_->sample_freq()==0) {
            Sk_->sample(w_gpu_);
        }
        // Save fields to file every save_freq_ steps
        if (it%P_->save_freq()==0) { 
            saveStdOutputFile("w_st_" + std::to_string(it));
            file_IO::saveGPUArray(w_gpu_+2*M_, "phi_st_" + std::to_string(it), 2*M_);
            Sk_->save("struct_st_"+ std::to_string(it));
        }
    }
    // Final save to file at end of equilibration period
    saveStdOutputFile("w_st_" + std::to_string(it-1));
    file_IO::saveGPUArray(w_gpu_+2*M_, "phi_st_" + std::to_string(it-1), 2*M_);
    Sk_->save("struct_st_"+ std::to_string(it-1));
}


// Calculate the diblock copolymer Hamiltonian
double lfts_simulation::getH() {
    // Calculate the natural log of the partition function
    double lnQ = dbc_->calc_concs(w_gpu_);

    // Create a Thrust device pointer to the GPU memory for the fields
    thrust::device_ptr<double> dp = thrust::device_pointer_cast(w_gpu_);

    // Calculate the sum of w+(r) and the sum of w-(r)^2 on the GPU
    double w_sum = thrust::reduce(dp+M_, dp+2*M_, 0.0, thrust::plus<double>());
    double w2_sum = thrust::transform_reduce(dp, dp+M_, thrust::square<double>(), 0.0, thrust::plus<double>());

    // Return the Hamiltonian
    return -lnQ + (w2_sum/P_->XbN() - w_sum)/M_;
}


// Save data in a standard format to be used as in input file
void lfts_simulation::saveStdOutputFile(std::string fileName) {
    P_->saveOutputParams(fileName);
    file_IO::saveGPUArray(w_gpu_, fileName, 2*M_, true);
}


// Generate an initial w-(r) field (sets w+(r) = 0)
void lfts_simulation::generate_field(double *w, int loadType) {
    switch (loadType) {
        case 2:
            field_generator::create_lamellar(w, P_->XbN(), P_->m());
            std::cout << "Generated lamellar initial configuration..." << std::endl;
            break;
        default:
            // Create a random field with noise of amplitude XN/2 
            // Using curand to keep a single, consistent RNG for reproducibilty
            double *rand_gpu, *rand_cpu;
            rand_cpu = new double[M_];
            GPU_ERR(cudaMalloc((void**)&rand_gpu, M_*sizeof(double)));
            curandGenerateUniformDouble(RNG_, rand_gpu, M_);
            GPU_ERR(cudaMemcpy(rand_cpu,rand_gpu, M_*sizeof(double), cudaMemcpyDeviceToHost));
            for (int r=0; r<M_; r++) {
                w[r] = P_->XbN()*(rand_cpu[r]-0.5);
                w[r+M_] = 0.0;
            }
            delete[] rand_cpu;
            GPU_ERR(cudaFree(rand_gpu));
            std::cout << "Generated disordered initial configuration..." << std::endl;
            break;
    }
}
        
