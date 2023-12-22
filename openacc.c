#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define M 1024
#define N 2048

void matrixVectorMultiplyWithReLU(int *matrix, int *vector, int *result) {
    #pragma acc kernels present(matrix[0:M*N], vector[0:N], result[0:M])
    for (int i = 0; i < M; i++) {
        result[i] = 0;
        #pragma acc loop reduction(+:result[i])
        for (int j = 0; j < N; j++) {
            result[i] += matrix[i * N + j] * vector[j];
        }
        // Apply ReLU activation
        result[i] = (result[i] > 0) ? result[i] : 0;
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

    startTime = clock();

    // Perform matrix-vector multiplication with ReLU activation using OpenACC
    #pragma acc data copyin(parallelMatrix[0:M*N], parallelVector[0:N]) \
                     copyout(parallelResult[0:M])
    {
        matrixVectorMultiplyWithReLU(parallelMatrix, parallelVector, parallelResult);
    }
    
    #pragma acc wait

    endTime = clock();
    double parallelTime = (double)(endTime - startTime) / CLOCKS_PER_SEC;
    printf("Parallel execution time: %f seconds\n", parallelTime);

    // Cleanup
    free(parallelMatrix);
    free(parallelVector);
    free(parallelResult);

    return 0;
}
