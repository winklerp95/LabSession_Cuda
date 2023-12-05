#pragma once

__global__
void matrixMulKernel(const float* A, const float* B, float* C,
    int numElements);
