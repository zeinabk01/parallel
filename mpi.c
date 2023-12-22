#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

#define M 1024
#define N 2048

// Function to perform matrix-vector multiplication with ReLU activation for a local portion
void matrixVectorMultiplyWithReLU(int *localMatrix, int *vector, int *localResult, int localSize) {
    // Loop over rows of the local matrix
    for (int i = 0; i < localSize; i++) {
        localResult[i] = 0;
        // Loop over columns of the matrix (assuming row-major order)
        for (int j = 0; j < N; j++) {
            // Multiply corresponding elements and accumulate the result
            localResult[i] += localMatrix[i * N + j] * vector[j];
        }
        // Apply ReLU activation
        localResult[i] = (localResult[i] > 0) ? localResult[i] : 0;
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
    double startTime, endTime, parallelTime;

    // Parallel version timing and computation
    MPI_Barrier(MPI_COMM_WORLD);
    startTime = MPI_Wtime();

    // Calculate local size and starting row for each process
    int localSize = M / world_size;
    int startRow = rank * localSize;

    // Allocate memory for local matrix, vector, and local result
    int *localMatrix = (int *)malloc(localSize * N * sizeof(int));
    int *vector = (int *)malloc(N * sizeof(int));
    int *localResult = (int *)malloc(localSize * sizeof(int));

    // Initialize local matrix and vector concurrently
    for (int i = 0; i < localSize * N; i++) {
        localMatrix[i] = startRow + 1;
    }

    for (int i = 0; i < N; i++) {
        vector[i] = 1;
    }

    // Perform local matrix-vector multiplication with ReLU activation
    matrixVectorMultiplyWithReLU(localMatrix, vector, localResult, localSize);

    // Gather local results to the root process
    int *gatheredResults = NULL;
    if (rank == 0) {
        gatheredResults = (int *)malloc(M * sizeof(int));
    }
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
    if (rank == 0) {
        free(gatheredResults);
    }

    // Finalize MPI
    MPI_Finalize();

    return EXIT_SUCCESS;
}
