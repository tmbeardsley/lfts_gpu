#!/bin/bash
#$ -cwd

mkdir ./build

nvcc ./src/fts_gpu.cu ./src/diblock.cu ./src/GPUkernels.cu ./src/GPUerror.cu ./src/step.cu ./src/anderson.cu ./src/field_generator.cc ./src/file_IO.cu ./src/langevin.cu ./src/strFunc.cu ./src/lfts_params.cc ./src/lfts_simulation.cu -o ./build/lfts-gpu -O3 -lcufft -lgsl -lgslcblas -lcurand -gencode=arch=compute_52,code=sm_52 -gencode=arch=compute_60,code=sm_60 -gencode=arch=compute_61,code=sm_61 -gencode=arch=compute_70,code=sm_70 -gencode=arch=compute_75,code=sm_75 -gencode=arch=compute_80,code=sm_80 -gencode=arch=compute_80,code=compute_80
