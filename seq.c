#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define M 1024
#define N 2048

void matrixVectorMultiply(int *matrix, int *vector, int *result) {
    for (int i = 0; i < M; i++) {
        result[i] = 0;
        for (int j = 0; j < N; j++) {
            result[i] += matrix[i * N + j] * vector[j];
        }
    }
}

int main() {
    double startTime, endTime;

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

    startTime = clock();

    // Perform matrix-vector multiplication in serial mode
    matrixVectorMultiply(serialMatrix, serialVector, serialResult);

    endTime = clock();
    double serialTime = (double)(endTime - startTime) / CLOCKS_PER_SEC;
    printf("Serial execution time: %f seconds\n", serialTime);

   

    // Cleanup
    free(serialMatrix);
    free(serialVector);
    free(serialResult);

    return 0;
}
