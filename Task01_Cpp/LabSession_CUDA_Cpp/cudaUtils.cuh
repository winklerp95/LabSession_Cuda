#pragma once
#include <vector>
#include <string>
#include "cuda_runtime.h"

using namespace std;

vector<string> runCudaGetDevices();
void checkError(cudaError_t err, const char* operation);