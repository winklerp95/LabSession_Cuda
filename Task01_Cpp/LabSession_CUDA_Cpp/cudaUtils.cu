#include <stdio.h>
#include <iostream>
#include <vector>
#include <sstream>

// For the CUDA runtime routines (prefixed with "cuda_")
#include "cuda_runtime.h"
#include "./cudaUtils.cuh"

using namespace std;

void checkError(cudaError_t err, const char* operation) {
    if (err != cudaSuccess) {
        std::cerr << "Error during " << operation << ": " << err << std::endl;
        std::cerr << cudaGetErrorString(err);
        exit(err);
    }
}

vector<string> runCudaGetDevices() {

    vector<string> deviceInfoVector;

    int deviceCount;
    cudaGetDeviceCount(&deviceCount);

    if (deviceCount == 0) {
        cout << "No CUDA-capable devices found";
    }
    else {
        cout << "Devices found: " << endl;

        for (int deviceID = 0; deviceID < deviceCount; ++deviceID) {
            cudaDeviceProp deviceProp;
            cudaGetDeviceProperties(&deviceProp, deviceID);

            deviceInfoVector.push_back(deviceProp.name);
            cout << "  " << deviceProp.name << endl;
        }
    }

    return deviceInfoVector;
}