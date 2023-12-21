%%writefile cuda_code.cu
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>

#define M 1024
#define N 2048
#define BLOCK_SIZE 256
#define TILE_SIZE 16  // Size of the matrix tile loaded into shared memory

__global__ void matrixVectorMultiply(int *matrix, int *vector, int *result, int m, int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    __shared__ int sharedVector[BLOCK_SIZE];
    __shared__ int sharedMatrixTile[TILE_SIZE][TILE_SIZE];

    // Load vector into shared memory
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        sharedVector[i] = vector[i];
    }

    __syncthreads();

    // Process multiple elements per thread
    for (int i = tid; i < m; i += stride) {
        int sum = 0;

        // Load matrix tile into shared memory
        for (int j = threadIdx.x; j < TILE_SIZE; j += blockDim.x) {
            sharedMatrixTile[j][threadIdx.x] = matrix[i * n + j * blockDim.x + threadIdx.x];
        }

        __syncthreads();

        // Compute dot product using the matrix tile in shared memory
        for (int j = 0; j < TILE_SIZE; j++) {
            sum += sharedMatrixTile[j][threadIdx.x] * sharedVector[j];
        }

        // Store the result
        result[i] = sum;
    }
}

int main() {
    int *h_matrix, *h_vector, *h_result;
    int *d_matrix, *d_vector, *d_result;

    h_matrix = (int *)malloc(M * N * sizeof(int));
    h_vector = (int *)malloc(N * sizeof(int));
    h_result = (int *)malloc(M * sizeof(int));

    // Initialize host matrix and vector

    // ... (your existing code)

    cudaMalloc((void **)&d_matrix, M * N * sizeof(int));
    cudaMalloc((void **)&d_vector, N * sizeof(int));
    cudaMalloc((void **)&d_result, M * sizeof(int));

    cudaMemcpy(d_matrix, h_matrix, M * N * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_vector, h_vector, N * sizeof(int), cudaMemcpyHostToDevice);

    dim3 gridDim((M + BLOCK_SIZE - 1) / BLOCK_SIZE, 1, 1);
    dim3 blockDim(BLOCK_SIZE, 1, 1);

    struct timeval start, end;
    gettimeofday(&start, NULL);

    // Launch the kernel
    matrixVectorMultiply<<<gridDim, blockDim>>>(d_matrix, d_vector, d_result, M, N);
    cudaDeviceSynchronize();

    gettimeofday(&end, NULL);

    double parallelTime = (end.tv_sec - start.tv_sec) + (end.tv_usec - start.tv_usec) / 1.0e6;

    printf("Parallel execution time: %f seconds\n", parallelTime);

    // ... (your existing code)

    free(h_matrix);
    free(h_vector);
    free(h_result);
    cudaFree(d_matrix);
    cudaFree(d_vector);
    cudaFree(d_result);

    return 0;
}
