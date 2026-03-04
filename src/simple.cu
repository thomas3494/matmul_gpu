#include <stdio.h>
#include <stdlib.h>
#include <cublas_v2.h>
#include <debug.h>
#include <test_code.h>
#include <assert.h>

#define MR 8
#define NR 8

__device__
void outer_product(float c[MR][NR], float a[MR], float b[NR])
{
    for (int i = 0; i < MR; i++) {
        for (int j = 0; j < NR; j++) {
            c[i][j] += a[i] * b[j];
        }
    }
}

/**
 * Duplicates column of a matrix across all rows in a 2D thread grid.
 **/
__device__
void bcast_col(int m , float *reg /* [m]                  */,
               int ld, float *x   /* [m * blockDim.y, ld] */)
{
    for (int i = 0; i < m; i++) {
        reg[i] = x[(blockDim.y * i + threadIdx.y) * ld];
    }
}

/**
 * Duplicates array across all columns in a 2D thread grid.
 **/
__device__
void bcast_row(int m , float *reg  /* [m]              */,
               float *x            /* [m * blockDim.x] */)
{
    for (int i = 0; i < m; i++) {
        reg[i] = x[blockDim.x * i + threadIdx.x];
    }
}

__global__
void matmul_kernel(float *c, float *a, float *b, int m, int k, int n)
{
    assert(gridDim.y * blockDim.y * MR == m);
    assert(gridDim.x * blockDim.x * NR == n);
    assert(k % KB == 0);

    float c_reg[MR][NR] = {(float)0};
    float a_reg[MR];
    float b_reg[NR];

    a += blockIdx.y * blockDim.y * MR * k;
    b += blockIdx.x * blockDim.x * NR;
    c += blockIdx.y * blockDim.y * MR * n + blockIdx.x * blockDim.x * NR;

    for (int p = 0; p < k; p++) {
        bcast_col(MR, a_reg, k, a + p    );
        bcast_row(NR, b_reg,    b + p * n);
        outer_product(c_reg, a_reg, b_reg);
    }

    for (int i = 0; i < MR; i++) {
        for (int j = 0; j < NR; j++) {
            int glob_i = i * blockDim.y + threadIdx.y;
            int glob_j = j * blockDim.x + threadIdx.x;
            c[glob_i * n + glob_j] = c_reg[i][j];
        }
    }
}

void matmul(float *c, float *a, float *b, int m, int k, int n)
{
    int threads_x = 32; // n
    int threads_y = 16; // m

    dim3 threadsPerBlock(threads_x, threads_y);
    dim3 blocks(n / (NR * threads_x), m / (MR * threads_y));
    
    matmul_kernel<<<blocks, threadsPerBlock>>>(c, a, b, m, k, n);
    CUDA_SAFE(cudaDeviceSynchronize());
    CUDA_SAFE(cudaPeekAtLastError());
}

int main(int argc, char **argv)
{
    if (argc != 4) {
        printf("Usage: M K N\n");
        return EXIT_FAILURE;
    }

    int m = atoi(argv[1]);
    int k = atoi(argv[2]);
    int n = atoi(argv[3]);

    float *d_a, *d_b, *d_c;
    init_all(&d_c, &d_a, &d_b, m, k, n);

    float duration;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventRecord(start, 0);

    matmul(d_c, d_a, d_b, m, k, n);

    cudaEventCreate(&stop);
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&duration, start,stop);

    duration /= 1e3;

    fprintf(stderr, "Gflops/s: ");
    printf("%f\n", 2.0 * m * n * k / duration / 1e9);

    check(d_c, m, k, n);

    CUDA_SAFE(cudaFree(d_a));
    CUDA_SAFE(cudaFree(d_b));
    CUDA_SAFE(cudaFree(d_c));

    return EXIT_SUCCESS;
}
