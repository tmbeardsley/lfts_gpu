#pragma once

#include <cufft.h>
#include <iostream>

struct cufftDeleter {
    using pointer = cufftHandle*;
    void operator()(pointer plan) const { 
        if (plan) {
            cufftDestroy(*plan);    // free gpu resources associated with the cufft plan.
            delete plan;            // free memory allocated for the cufftHandle.
        }
    }
};