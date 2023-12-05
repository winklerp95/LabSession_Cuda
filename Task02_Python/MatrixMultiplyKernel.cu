#include <torch/extension.h>
#include <cuda_runtime.h>

__global__
void matrixMulKernel(const float* A, const float* B, float* Result, int numElements) {
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

/**
 * Multiplies two matrices using CUDA.
 *
 * This function takes two input matrices A and B, and computes their matrix
 * multiplication using a CUDA kernel. The result is stored in the output matrix
 * Result.
 *
 * Parameters:
 *   A_tensor (torch::Tensor): Input matrix A.
 *   B_tensor (torch::Tensor): Input matrix B.
 *   Result_tensor (torch::Tensor): Output matrix Result.
 *   numElements (int): Number of elements in each dimension of the matrices.
 *
 * Usage:
 *   A = torch.rand(matrix_size, matrix_size).cuda()
 *   B = torch.rand(matrix_size, matrix_size).cuda()
 *   Result = MatrixMultiply(A, B, matrix_size)
 */
void MatrixMultiply(const torch::Tensor& A_tensor, const torch::Tensor& B_tensor, torch::Tensor& Result_tensor, int numElements) {

    // Check if the tensors are on the GPU
    if (!A_tensor.is_cuda() || !B_tensor.is_cuda() || !Result_tensor.is_cuda()) {
        throw std::runtime_error("All tensors must be on the GPU.");
    }

    // Get value pointers from tensor objects
    const float* A = A_tensor.data_ptr<float>();
    const float* B = B_tensor.data_ptr<float>();
    float* Result = Result_tensor.data_ptr<float>();

    // Get Block and Grid size
    dim3 threadsPerBlock(numElements, numElements);
    dim3 blocksPerGrid(1, 1);
    if (numElements * numElements > 256) {
        threadsPerBlock.x = 16;
        threadsPerBlock.y = 16;
        blocksPerGrid.x = ceil(double(numElements) / double(threadsPerBlock.x));
        blocksPerGrid.y = ceil(double(numElements) / double(threadsPerBlock.y));
    }

    // Launch kernel
    matrixMulKernel<<<blocksPerGrid, threadsPerBlock>>>(A, B, Result, numElements);

    // Check error
    auto err = cudaGetLastError();
    if (cudaSuccess != err) {
        fprintf(stderr, "CUDA kernel failed : %s\n", cudaGetErrorString(err));
        exit(-1);
    }
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("MatrixMultiply", &MatrixMultiply, "Matrix multiplication");
}
