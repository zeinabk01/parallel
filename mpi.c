#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

#define M 1024
#define N 2048

// Function to perform matrix-vector multiplication for a local portion
void matrixVectorMultiply(int *localMatrix, int *vector, int *localResult, int localSize) {
    // Loop over rows of the local matrix
    for (int i = 0; i < localSize; i++) {
        localResult[i] = 0;
        // Loop over columns of the matrix (assuming row-major order)
        for (int j = 0; j < N; j++) {
            // Multiply corresponding elements and accumulate the result
            localResult[i] += localMatrix[i * N + j] * vector[j];
        }
    }
}

int main(int argc, char **argv) {
    // Initialize MPI
    MPI_Init(&argc, &argv);

    // Get MPI information: rank and total number of processes
    int world_size, rank;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    // Timing variables
    double startTime, endTime, serialTime, parallelTime;

    // Serial version timing and computation (only executed by rank 0)
    if (rank == 0) {
        startTime = MPI_Wtime();

        // Allocate memory for serial matrix, vector, and result
        int *serialMatrix = (int *)malloc(M * N * sizeof(int));
        int *serialVector = (int *)malloc(N * sizeof(int));
        int *serialResult = (int *)malloc(M * sizeof(int));

        // Initialize serial matrix and vector (you may replace this with your own initialization logic)
        for (int i = 0; i < M * N; i++) {
            serialMatrix[i] = 1;
        }

        for (int i = 0; i < N; i++) {
            serialVector[i] = 1;
        }

        // Perform serial matrix-vector multiplication
        matrixVectorMultiply(serialMatrix, serialVector, serialResult, M);

        // Free memory for serial version
        free(serialMatrix);
        free(serialVector);
        free(serialResult);

        // Measure and print serial execution time
        endTime = MPI_Wtime();
        serialTime = endTime - startTime;
        printf("Serial execution time: %f seconds\n", serialTime);
    }

    // Parallel version timing and computation
    MPI_Barrier(MPI_COMM_WORLD);
    startTime = MPI_Wtime();

    // Calculate local size for each process
    int localSize = M / world_size;

    // Allocate memory for local matrix, vector, local result, and gathered results
    int *localMatrix = (int *)malloc(localSize * N * sizeof(int));
    int *vector = (int *)malloc(N * sizeof(int));
    int *localResult = (int *)malloc(localSize * sizeof(int));
    int *gatheredResults = (int *)malloc(M * sizeof(int));

    // Initialize local matrix and vector (you may replace this with your own initialization logic)
    for (int i = 0; i < localSize * N; i++) {
        localMatrix[i] = rank + 1;
    }

    for (int i = 0; i < N; i++) {
        vector[i] = 1;
    }

    // Perform local matrix-vector multiplication
    matrixVectorMultiply(localMatrix, vector, localResult, localSize);

    // Gather local results to the root process
    MPI_Gather(localResult, localSize, MPI_INT, gatheredResults, localSize, MPI_INT, 0, MPI_COMM_WORLD);

    // Free memory for parallel version
    free(localMatrix);
    free(vector);
    free(localResult);

    // Synchronize processes before measuring end time
    MPI_Barrier(MPI_COMM_WORLD);
    endTime = MPI_Wtime();

    // Print total execution time on the root process
    if (rank == 0) {
        parallelTime = endTime - startTime;
        printf("Total execution time: %f seconds\n", parallelTime);
    }

    // Free memory for gathered results
    free(gatheredResults);

    // Finalize MPI
    MPI_Finalize();

    return EXIT_SUCCESS;
}
