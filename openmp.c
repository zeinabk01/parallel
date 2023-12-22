#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

#define M 1024
#define N 2048

// Function to perform matrix-vector multiplication with ReLU activation in parallel using OpenMP
void matrixVectorMultiplyWithReLU(int *matrix, int *vector, int *result) {
    #pragma omp parallel for
    for (int i = 0; i < M; i++) {
        result[i] = 0;

        for (int j = 0; j < N; j++) {
            result[i] += matrix[i * N + j] * vector[j];
        }

        // Apply ReLU activation
        result[i] = (result[i] > 0) ? result[i] : 0;
    }
}

int main() {
    double startTime, endTime;

    // Perform matrix-vector multiplication with ReLU activation in serial mode
    int *serialMatrix = (int *)malloc(M * N * sizeof(int));
    int *serialVector = (int *)malloc(N * sizeof(int));
    int *serialResult = (int *)malloc(M * sizeof(int));

    // Initialize serialMatrix and serialVector 
    for (int i = 0; i < M * N; i++) {
        serialMatrix[i] = 1; 
    }

    for (int i = 0; i < N; i++) {
        serialVector[i] = 1;  
    }

    startTime = omp_get_wtime(); // Start timing

    // Perform matrix-vector multiplication with ReLU activation in parallel using OpenMP
    matrixVectorMultiplyWithReLU(serialMatrix, serialVector, serialResult);

    endTime = omp_get_wtime(); // Stop timing
    double parallelTime = endTime - startTime; // Calculate execution time
    printf("Parallel execution time: %f seconds\n", parallelTime);

    // Cleanup
    free(serialMatrix);
    free(serialVector);
    free(serialResult);

    return 0;
}
