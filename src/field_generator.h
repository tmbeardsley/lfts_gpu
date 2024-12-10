// #############################################################################
// Provides functions for creating input fields
// #############################################################################
#pragma once
#include <cmath>

namespace field_generator {

    // Create a lamellar w-(r) field with a cosine wave of size Amplitude.
    // (kx, ky, kz) is the wavevector defining the lamellar orientation. m[] is the mesh size.
    void create_lamellar(double *w, double Amplitude, int *m, int kx=3, int ky=0, int kz=0);
    
}