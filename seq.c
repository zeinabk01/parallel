#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define M 1024
#define N 2048

void matrixVectorMultiplyWithReLU(int *matrix, int *vector, int *result) {
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

    startTime = clock();

    // Perform matrix-vector multiplication with ReLU activation in serial mode
    matrixVectorMultiplyWithReLU(serialMatrix, serialVector, serialResult);

    endTime = clock();
    double serialTime = (double)(endTime - startTime) / CLOCKS_PER_SEC;
    printf("Serial execution time: %f seconds\n", serialTime);

    // Cleanup
    free(serialMatrix);
    free(serialVector);
    free(serialResult);

    return 0;
}
