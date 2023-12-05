/**
 * Matrix multiplication: C = A * B.
 *     where A, B and C are NxN matrices
 *
 * This sample is a very basic sample that implements a matrix multiplication
 */

// Default includes
#include <stdio.h>
#include <chrono>
#include <iostream>
#include <string>
#include <cstdlib>

 // For the CUDA runtime routines
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <curand_mtgp32_kernel.h>
#include <cuda.h>

// Custom inclues
#include "./runGPU.cuh"
#include "./cudaUtils.cuh"
#include "./TimeTracker.h"

using namespace std;

// Matrix multiplication CUDA kernel
__global__
void matrixMulKernel(const float* A, const float* B, float* Result,
    int numElements) {

    int ROW = blockIdx.y * blockDim.y + threadIdx.y;
    int COL = blockIdx.x * blockDim.x + threadIdx.x;

    float tmpSum = 0;

    if (ROW < numElements && COL < numElements) {
        // each thread computes one element of the block sub-matrix
        for (int i = 0; i < numElements; i++) {
            tmpSum += A[ROW * numElements + i] * B[i * numElements + COL];
        }

        Result[ROW * numElements + COL] = tmpSum;
    }

}

long runGPU(float* h_A, float* h_B, float* h_Result, int numElements) {

    // Check memory allocation
    if (h_A == nullptr || h_B == nullptr || h_Result == nullptr) {
        cout << "Memory allocation failure!";
        return -1;
    }

    // Get Devices
    auto devices = runCudaGetDevices();
    if (devices.empty()) {
        cout << "No CUDA GPU found!";
        return -1;
    }

    // Get size of matrix
    size_t size = numElements * numElements * sizeof(float);

    // Error code to check return values for CUDA calls
    cudaError_t err = cudaSuccess;

    // Start time tracking
    TimeTracker tracker("GPU");

    // Allocate the device input matrix A
    float* d_A = NULL;
    // TODO: Allocate matrix A on cuda device, e.g. err = cudaMalloc(...)
    // Your Code here
    
    checkError(err, "cudaMalloc d_A");

    // Allocate the device input matrix B
    float* d_B = NULL;
    // TODO: Allocate matrix B on cuda device, e.g. err = cudaMalloc(...)
    // Your Code here
    
    checkError(err, "cudaMalloc d_B");

    // Allocate the device output matrix C
    float* d_Result = NULL;
    // TODO: Allocate matrix Result on cuda device, e.g. err = cudaMalloc(...)
    // Your Code here
    
    checkError(err, "cudaMalloc d_C");

    // Copy the host input to device memory     
    // TODO: Copy Matrix A from Host to Device e.g. err = cudaMemcpy(...)
    // Your Code here
    
    checkError(err, "cudaMemcpyHostToDevice (A)");

    // TODO: Copy Matrix B from Host to Device e.g. err = cudaMemcpy(...)
    // Your Code here
    
    checkError(err, "cudaMemcpyHostToDevice (B)");

    // Get Block and Grid size
    dim3 threadsPerBlock;
    dim3 blocksPerGrid;
    if (numElements * numElements <= 256) {
        dim3 threadsPerBlock(numElements, numElements);
        dim3 blocksPerGrid(1, 1);
    }
    else {
        threadsPerBlock.x = 16;
        threadsPerBlock.y = 16;
        blocksPerGrid.x = ceil(float(numElements) / float(threadsPerBlock.x));
        blocksPerGrid.y = ceil(float(numElements) / float(threadsPerBlock.y));
    }

    // TODO: Launch the Kernel e.g. matrixMulKernel ... (...);
    // Your Code here
    
    err = cudaGetLastError();
    checkError(err, "Kernel Launch");

    // Wait for Sync
    cudaDeviceSynchronize();

    // Copy the device resultto the host result
    // TODO: Copy Matrix C from Host to Device e.g. err = cudaMemcpy(...)
    // Your Code here
    
    checkError(err, "Kernel Launch");

    // Free device global memory    
    // TODO: Free the allocated memory for Matrix A e.g. err = cudaFree(...)
    // Your Code here
    
    checkError(err, "cudaFree(d_A)");

    // TODO: Free the allocated memory for Matrix B e.g. err = cudaFree(...)
    // Your Code here
    
    checkError(err, "cudaFree(d_B)");

    // TODO: Free the allocated memory for Matrix C e.g. err = cudaFree(...)
    // Your Code here
    
    checkError(err, "cudaFree(d_C)");

    // Store duration
    long duration = tracker.stop();

    return duration;
}

