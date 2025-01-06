// #######################################################################
// Creates a new lfts_simulation(...) object, passing in input file to 
// its contructor and subsequently accessing its public methods to 
// equilibrate the system, gather statistics from the equilibrated system
// and finally output the system's energy
// #######################################################################

#include <string>
#include <iostream>
#include "lfts_simulation.h"
#include <memory>
#include <chrono>
#include <memory>

#ifdef VIEW_FIELD
    // Include classes and libraries necessary for visualisation
    #include "viewFieldClass.h"
    #include <thread>
#endif

void simulation_loop(lfts_simulation*);

using namespace std;



int main (int argc, char *argv[])
{
    // Get input file name from command-line argument
    if (argc != 2) {
        cout << "Please supply a single input file name as a command-line argument." << endl << endl;
        exit(1);
    }
    string inputFile(argv[1]);

    // New lfts_simulation object with input file name specified
    // and 512 threads per block on the gpu
    unique_ptr<lfts_simulation> lfts_sim = make_unique<lfts_simulation>(inputFile, 512);

    #ifdef VIEW_FIELD
        // Create instance of viewFieldClass for visualisation
        unique_ptr<viewFieldClass> viewField = make_unique<viewFieldClass>(lfts_sim->P_->m());

        // Give the main simulation controller access to the viewFieldClass instance to pass data to it
        lfts_sim->set_field_viewer(viewField.get());

        // Start the main simulation on a new thread
        std::thread equilThread = std::thread([&]() {
            simulation_loop(lfts_sim.get());
        });

        // Start the interactor of viewFieldClass to observe the field
        viewField->startInteractor();

        // Wait for the interaction thread to finish
        equilThread.join(); 
    #else
        simulation_loop(lfts_sim.get());
    #endif
    
    

    // Output the final energy of the system
    cout.precision(6);
    cout << lfts_sim->getH() << endl;

    return 0;
}


// Perform and time the equilibration and statistics portion of the simulation
void simulation_loop(lfts_simulation* lfts_sim) {

        cout << "Starting Equilibration..." << endl;
        auto start = std::chrono::steady_clock::now();
        lfts_sim -> equilibrate();
        auto end = std::chrono::steady_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::seconds>(end - start);
        cout << "Equlibration time = " << duration.count() << "secs" << endl << endl;

        // Time the statistics gathering period
        cout << "Starting Statistics..." << endl;
        start = std::chrono::steady_clock::now();
        lfts_sim -> statistics();
        end = std::chrono::steady_clock::now();
        duration = std::chrono::duration_cast<std::chrono::seconds>(end - start);
        cout << "Statistics time = " << duration.count() << "secs" << endl << endl;
}