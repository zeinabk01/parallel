#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>

#define M 1024
#define N 2048

// OpenACC kernel for matrix-vector multiplication
#pragma acc routine seq
void matrixVectorMultiply(int *matrix, int *vector, int *result, int m, int n) {
    #pragma acc loop seq
    for (int i = 0; i < m; i++) {
        int sum = 0;
        #pragma acc loop seq
        for (int j = 0; j < n; j++) {
            sum += matrix[i * n + j] * vector[j];
        }
        result[i] = sum;
    }
}

int main() {
    int *h_matrix, *h_vector, *h_result;
    int *d_matrix, *d_vector, *d_result;

    // Allocate host memory
    h_matrix = (int *)malloc(M * N * sizeof(int));
    h_vector = (int *)malloc(N * sizeof(int));
    h_result = (int *)malloc(M * sizeof(int));

    // Initialize host matrix and vector (you may replace this with your own initialization logic)
    for (int i = 0; i < M * N; i++) {
        h_matrix[i] = 1;  // Example initialization
    }

    for (int i = 0; i < N; i++) {
        h_vector[i] = 1;  // Example initialization
    }

    // Allocate device memory
    d_matrix = (int *)malloc(M * N * sizeof(int));
    d_vector = (int *)malloc(N * sizeof(int));
    d_result = (int *)malloc(M * sizeof(int));

    // Copy data from host to device
    #pragma acc enter data copyin(h_matrix[0:M*N], h_vector[0:N]) create(d_matrix[0:M*N], d_vector[0:N], d_result[0:M])

    // Copy data from host to device
    #pragma acc update device(d_matrix[0:M*N], d_vector[0:N])

    // Parallel version timing
    struct timeval start, end;
    gettimeofday(&start, NULL);

    // OpenACC offloading
    #pragma acc parallel present(d_matrix[0:M*N], d_vector[0:N], d_result[0:M])
    {
        matrixVectorMultiply(d_matrix, d_vector, d_result, M, N);
    }

    gettimeofday(&end, NULL);

    double parallelTime = (end.tv_sec - start.tv_sec) + (end.tv_usec - start.tv_usec) / 1.0e6;

    printf("Parallel execution time: %f seconds\n", parallelTime);

    // ... (your existing code)

    // Free host and device memory
    #pragma acc exit data delete(h_matrix[0:M*N], h_vector[0:N], d_matrix[0:M*N], d_vector[0:N], d_result[0:M])
    free(h_matrix);
    free(h_vector);
    free(h_result);
    free(d_matrix);
    free(d_vector);
    free(d_result);

    return 0;
}
