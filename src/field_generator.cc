// #############################################################################
// Provides functions for creating input fields
// #############################################################################

#include "field_generator.h"

namespace field_generator {

    // Create a lamellar w-(r) field with a cosine wave of size Amplitude.
    // (kx, ky, kz) is the wavevector defining the lamellar orientation. m[] is the mesh size.
    void create_lamellar(double *w, double Amplitude, int *m, int kx, int ky, int kz) {
        int r, M;

        M = m[0]*m[1]*m[2];
        for (int mx=0; mx<m[0]; mx++) {
            for (int my=0; my<m[1]; my++) {
                for (int mz=0; mz<m[2]; mz++) {
                    r = m[2] * (mx*m[1] + my) + mz;	    // Row Major Indexing
                    w[r] = Amplitude * std::cos(2.0*M_PI*(  kx*static_cast<double>(mx)/m[0] + 
                                                            ky*static_cast<double>(my)/m[1] + 
                                                            kz*static_cast<double>(mz)/m[2]));
                    w[r+M] = 0.0;
                }
            }
        }
    }
}