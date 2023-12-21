#include <stdio.h>
#include <stdlib.h>
#include <omp.h>


#define M 1024
#define N 2048

void matrixVectorMultiply(int *matrix, int *vector, int *result) {
    #pragma omp parallel for
    for (int i = 0; i < M; i++) {
        result[i] = 0;
        for (int j = 0; j < N; j++) {
            result[i] += matrix[i * N + j] * vector[j];
        }
    }
}

int main() {
    double startTime, endTime;

    // Perform matrix-vector multiplication in serial (non-parallel) mode
    int *serialMatrix = (int *)malloc(M * N * sizeof(int));
    int *serialVector = (int *)malloc(N * sizeof(int));
    int *serialResult = (int *)malloc(M * sizeof(int));

    // Initialize serialMatrix and serialVector (you may replace this with your own initialization logic)
    for (int i = 0; i < M * N; i++) {
        serialMatrix[i] = 1;  // Example initialization
    }

    for (int i = 0; i < N; i++) {
        serialVector[i] = 1;  // Example initialization
    }

    startTime = omp_get_wtime();

    // Perform matrix-vector multiplication in serial mode
    matrixVectorMultiply(serialMatrix, serialVector, serialResult);

    endTime = omp_get_wtime();
    double serialTime = endTime - startTime;  // Assign the value to serialTime
    printf("Serial execution time: %f seconds\n", serialTime);

    // Perform matrix-vector multiplication in parallel using OpenMP
    int *parallelMatrix = (int *)malloc(M * N * sizeof(int));
    int *parallelVector = (int *)malloc(N * sizeof(int));
    int *parallelResult = (int *)malloc(M * sizeof(int));

    // Initialize parallelMatrix and parallelVector (you may replace this with your own initialization logic)
    for (int i = 0; i < M * N; i++) {
        parallelMatrix[i] = 1;  // Example initialization
    }

    for (int i = 0; i < N; i++) {
        parallelVector[i] = 1;  // Example initialization
    }

    startTime = omp_get_wtime();

    // Perform matrix-vector multiplication in parallel using OpenMP
    matrixVectorMultiply(parallelMatrix, parallelVector, parallelResult);

    endTime = omp_get_wtime();
    double parallelTime = endTime - startTime;  // Assign the value to parallelTime
    printf("Parallel execution time: %f seconds\n", parallelTime);

    

    // Cleanup
    free(serialMatrix);
    free(serialVector);
    free(serialResult);

    free(parallelMatrix);
    free(parallelVector);
    free(parallelResult);

    return 0;
}
