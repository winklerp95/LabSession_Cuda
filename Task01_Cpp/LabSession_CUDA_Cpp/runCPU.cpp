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

// Custom inclues
#include "./TimeTracker.h"
#include "./runCPU.h"


using namespace std;


void matrixMul(const float* A, const float* B, float* C, int numElements) {
    for (int ROW = 0; ROW < numElements; ROW++) {
        for (int COL = 0; COL < numElements; COL++) {
            float tmpSum = 0;
            for (int i = 0; i < numElements; i++) {
                tmpSum += A[ROW * numElements + i] * B[i * numElements + COL];
            }
            C[ROW * numElements + COL] = tmpSum;
        }
    }
}


long runCPU(float* h_A, float* h_B, float* h_result, int numElements) {

    // Start time tracking
    TimeTracker tracker("CPU");

	// Check memory allocation
    if (h_A == nullptr || h_B == nullptr || h_result == nullptr) {
        cout << "Memory allocation failure";
        return -1;
    }

	// Run Matrix Multiplication
    matrixMul(h_A, h_B, h_result, numElements);

    // Store cleaning time
    long duration = tracker.stop();

    return duration;
}

