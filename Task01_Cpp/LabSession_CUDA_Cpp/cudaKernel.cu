/**
 * Matrix multiplication: C = A * B.
 *     where A, B and C are NxN matrices
 *
 * This sample is a very basic sample that implements a matrix multiplication
 */

#include <stdio.h>
#include <chrono>
#include <iostream>
#include <string>
#include <cstdlib>

 // For the CUDA runtime routines (prefixed with "cuda_")
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <curand_mtgp32_kernel.h>
#include <cuda.h>

// Own headers
#include "./cudaKernel.cuh"


using namespace std;


__global__
void matrixMulKernel(const float* A, const float* B, float* C,
    int numElements) {

    int ROW = blockIdx.y * blockDim.y + threadIdx.y;
    int COL = blockIdx.x * blockDim.x + threadIdx.x;

    float tmpSum = 0;

    if (ROW < numElements && COL < numElements) {
        // each thread computes one element of the block sub-matrix
        for (int i = 0; i < numElements; i++) {
            tmpSum += A[ROW * numElements + i] * B[i * numElements + COL];
        }

        C[ROW * numElements + COL] = tmpSum;
    }
    
}

