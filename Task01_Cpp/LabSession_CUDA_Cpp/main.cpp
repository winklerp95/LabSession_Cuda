#include <iostream>
#include <vector>
#include "./runGPU.cuh"
#include "./runCPU.h"


using namespace std;

int main(void)
{
	// ************************************************************************
    // Initialization
    // ************************************************************************
    int numElements = 1000;

    // Get size of matrix
    size_t size = numElements * numElements * sizeof(float);

    // Allocate memory on the host
    float* h_A = (float*)malloc(size);
    float* h_B = (float*)malloc(size);

    // Check memory allocation
    if (h_A == nullptr || h_B == nullptr ) {
        cout << "Memory allocation failure";
        return -1;
    }

    // Initialize input on the host
    for (int row = 0; row < numElements; row++) {
        for (int col = 0; col < numElements; col++) {
            h_A[row * numElements + col] = static_cast<float>(rand());
            h_B[row * numElements + col] = static_cast<float>(rand());
        }
    }


    // ************************************************************************
    // Execute test runs
    // ************************************************************************

    // Run Test on CPU (Sequential)
    float* resultCPU = (float*)malloc(size);
    long durationCPU = runCPU(h_A, h_B, resultCPU, numElements);

	// Run Test on GPU (Parallel)
    float* resultGPU = (float*)malloc(size);
    long durationGPU = runGPU(h_A, h_B, resultGPU, numElements);


    // ************************************************************************
    // Verification
    // ************************************************************************
    bool error;

    // Verify that the result matrix is correct
    if (resultCPU == nullptr || resultGPU == nullptr) {
        error = true;
    }
    else {
        error = false;
        for (int i = 0; i < numElements; i++) {
            for (int j = 0; j < numElements; j++) {
                if (resultCPU[i * numElements + j] != resultGPU[i * numElements + j]) {
                    cout << "Result verification failed at element: " << i << endl;
                    error = true;
                    break;
                }
            }
            if (error) { break; }
        }
    }
	
	// Print verification
    if (durationGPU == -1 || durationGPU == 0 || durationCPU == 0) {
        cout << "Your code is not correct.";
    }
	else if (error) {
		cout << "Verification error occurred.";
	}
    else {
        double performance = ((double)durationCPU / (double)durationGPU);
        cout << "The GPU is " << performance << " times faster." << endl;
    }

    // ************************************************************************
    // Cleanup / Finish
    // ************************************************************************

    // Free host memory
    free(h_A);
    free(h_B);
	free(resultCPU);
    free(resultGPU);

    // Wait for CLI interaction
    cin.get();

    return 0;
}
