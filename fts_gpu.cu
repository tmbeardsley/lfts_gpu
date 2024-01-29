// #######################################################################
// Creates a new lfts_simulation(...) object, passing in input file to 
// its contructor and subsequently accessing its public methods to 
// equilibrate the system, gather statistics from the equilibrated system
// and finally output the system's energy
// #######################################################################

#include <iostream>
#include "lfts_simulation.h"

using namespace std;

//------------------------------------------------------------
int main ()
{
    time_t t;

    // New lfts_simulation object with input file name specified
    lfts_simulation *lfts_sim = new lfts_simulation("input");
    
    // Time the equilibration period
    cout << "Starting Equilibrating..." << endl;
    t = time(NULL);
    lfts_sim -> equilibrate();
    cout << "Equlibration time = " << time(NULL)-t << "secs" << endl << endl;

    // Time the statistics gathering period
    cout << "Starting Statistics..." << endl;
    t = time(NULL);
    lfts_sim -> statistics();
    cout << "Statistics time = " << time(NULL)-t << "secs" << endl << endl;

    // Output the final energy of the system
    cout.precision(6);
    cout << lfts_sim->getH() << endl;

    delete lfts_sim;
    return 0;
}
