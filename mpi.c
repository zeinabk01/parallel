#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

#define M 1024
#define N 2048

void matrixVectorMultiply(int *localMatrix, int *vector, int *localResult, int localSize) {
    for (int i = 0; i < localSize; i++) {
        localResult[i] = 0;
        for (int j = 0; j < N; j++) {
            localResult[i] += localMatrix[i * N + j] * vector[j];
        }
    }
}

int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);

    int world_size, rank;

    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    double startTime, endTime, serialTime, parallelTime;

    // Serial version timing
    if (rank == 0) {
        startTime = MPI_Wtime();

        int *serialMatrix = (int *)malloc(M * N * sizeof(int));
        int *serialVector = (int *)malloc(N * sizeof(int));
        int *serialResult = (int *)malloc(M * sizeof(int));

        for (int i = 0; i < M * N; i++) {
            serialMatrix[i] = 1;
        }

        for (int i = 0; i < N; i++) {
            serialVector[i] = 1;
        }

        matrixVectorMultiply(serialMatrix, serialVector, serialResult, M);

        free(serialMatrix);
        free(serialVector);
        free(serialResult);

        endTime = MPI_Wtime();
        serialTime = endTime - startTime;
        printf("Serial execution time: %f seconds\n", serialTime);
    }

    // Parallel version timing
    MPI_Barrier(MPI_COMM_WORLD);
    startTime = MPI_Wtime();

    int localSize = M / world_size;
    int *localMatrix = (int *)malloc(localSize * N * sizeof(int));
    int *vector = (int *)malloc(N * sizeof(int));
    int *localResult = (int *)malloc(localSize * sizeof(int));
    int *gatheredResults = (int *)malloc(M * sizeof(int));

    // Initialize localMatrix and vector (you may replace this with your own initialization logic)
    for (int i = 0; i < localSize * N; i++) {
        localMatrix[i] = rank + 1;
    }

    for (int i = 0; i < N; i++) {
        vector[i] = 1;
    }

    matrixVectorMultiply(localMatrix, vector, localResult, localSize);

    MPI_Gather(localResult, localSize, MPI_INT, gatheredResults, localSize, MPI_INT, 0, MPI_COMM_WORLD);

    free(localMatrix);
    free(vector);
    free(localResult);

    MPI_Barrier(MPI_COMM_WORLD);
    endTime = MPI_Wtime();

    if (rank == 0) {
        parallelTime = endTime - startTime;
        printf("Total execution time: %f seconds\n", parallelTime);
    }

    free(gatheredResults);

    MPI_Finalize();

    return EXIT_SUCCESS;
}
