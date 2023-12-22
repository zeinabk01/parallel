#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define M 1024
#define N 2048
#define THREADS_PER_BLOCK 512
#define BLOCKS (M + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK

__global__ void matrixVectorMultiplyWithReLU(int *matrix, int *vector, int *result) {
    __shared__ int partialResult[THREADS_PER_BLOCK];

    int tid = threadIdx.x;
    int bid = blockIdx.x;
    int i = bid * THREADS_PER_BLOCK + tid;

    partialResult[tid] = 0;

    while (i < M) {
        int sum = 0;
        for (int j = 0; j < N; j++) {
            sum += matrix[i * N + j] * vector[j];
        }
        // Apply ReLU activation
        partialResult[tid] = (sum > 0) ? sum : 0;

        i += THREADS_PER_BLOCK * gridDim.x; // Move to the next block
    }

    __syncthreads();

    // Perform a parallel reduction to obtain the final result
    for (int stride = THREADS_PER_BLOCK / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            partialResult[tid] += partialResult[tid + stride];
        }
        __syncthreads();
    }

    // Write the final result to global memory
    if (tid == 0) {
        result[bid] = partialResult[0];
    }
}

int main() {
    clock_t startTime, endTime;

    int *parallelMatrix = (int *)malloc(M * N * sizeof(int));
    int *parallelVector = (int *)malloc(N * sizeof(int));
    int *parallelResult = (int *)malloc(M * sizeof(int));

    // Initialize parallelMatrix and parallelVector
    for (int i = 0; i < M * N; i++) {
        parallelMatrix[i] = 1;
    }

    for (int i = 0; i < N; i++) {
        parallelVector[i] = 1;
    }

    // Allocate device memory
    int *d_matrix, *d_vector, *d_result;
    cudaMalloc((void **)&d_matrix, M * N * sizeof(int));
    cudaMalloc((void **)&d_vector, N * sizeof(int));
    cudaMalloc((void **)&d_result, M * sizeof(int));

    // Copy data to device
    cudaMemcpy(d_matrix, parallelMatrix, M * N * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_vector, parallelVector, N * sizeof(int), cudaMemcpyHostToDevice);

    startTime = clock();

    // Perform matrix-vector multiplication with ReLU activation using CUDA
    matrixVectorMultiplyWithReLU<<<BLOCKS, THREADS_PER_BLOCK>>>(d_matrix, d_vector, d_result);

    // Synchronize to make sure the GPU has completed the computation
    cudaDeviceSynchronize();

    // Copy result back to host
    cudaMemcpy(parallelResult, d_result, M * sizeof(int), cudaMemcpyDeviceToHost);

    endTime = clock();
    double parallelTime = (double)(endTime - startTime) / CLOCKS_PER_SEC;
    printf("Parallel execution time: %f seconds\n", parallelTime);

    // Cleanup
    free(parallelMatrix);
    free(parallelVector);
    free(parallelResult);
    cudaFree(d_matrix);
    cudaFree(d_vector);
    cudaFree(d_result);

    return 0;
}
