#pragma once

#include <cuda.h>
#include <debug.h>
#include <math.h>
#include <curand_kernel.h>

__host__ __device__
int ceil(int a, int b)
{
    return (a + b - 1) / b;
}

struct Zero {
    __device__ __host__ float operator()(int, int) const {
        return 0.0f;
    }
};

template<typename F>
__global__
void init_d(float *x, int m, int n, F f)
{
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < m && j < n) {
        x[i * n + j] = f(i, j);
    }
}

__global__ 
void initRandomMatrix_kernel(float *matrix, int m, int n, unsigned long seed)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = m * n;

    if (idx < total) {
        // Initialize random state
        curandState state;
        curand_init(seed, idx, 0, &state);

        // Generate random float between 0 and 1
        matrix[idx] = curand_uniform(&state);
    }
}

void initRandomMatrix(float *matrix, int m, int n, unsigned long seed)
{
    int threads_x = 16;
    int threads_y = 16;
    dim3 threads(threads_x, threads_y);
    dim3 blocks (ceil(m, threads_x), ceil(n, threads_y));
    initRandomMatrix_kernel<<<blocks, threads>>>(matrix, m, n, seed);
}

template<typename F>
void init(float *x, int m, int n, F f)
{
    int threads_x = 16;
    int threads_y = 16;
    dim3 threads(threads_x, threads_y);
    dim3 blocks (ceil(m, threads_x), ceil(n, threads_y));
    
    init_d<<<blocks, threads>>>(x, m, n, f);

    CUDA_SAFE(cudaDeviceSynchronize());
    CUDA_SAFE(cudaPeekAtLastError());
}

void init_all(float **d_c, float **d_a, float **d_b,
              size_t m, size_t k, size_t n)
{
    CUDA_SAFE(cudaMalloc(d_a, m * k * sizeof(float)));
    CUDA_SAFE(cudaMalloc(d_b, k * n * sizeof(float)));
    CUDA_SAFE(cudaMalloc(d_c, m * n * sizeof(float)));

    initRandomMatrix(*d_a, m, k, 12390123ul);
    initRandomMatrix(*d_b, k, n, 192309123ul);
    init(*d_c, m, n, Zero());
}

float rel_error(float a, float b)
{
    return fabs(a - b) / b;
}

void check(float *d_c, float *d_a, float *d_b, size_t m, size_t k, size_t n)
{
    float *c  = (float *)malloc(m * n * sizeof(float));
    float *c2 = (float *)malloc(m * n * sizeof(float));
    CUDA_SAFE(cudaMemcpy(c, d_c, m * n * sizeof(float),
                cudaMemcpyDeviceToHost));

    cublasHandle_t handle;
    CUBLAS_SAFE(cublasCreate(&handle));
    const float alpha = 1.0;
    const float beta = 0.0;
    CUBLAS_SAFE(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k,
                            &alpha, d_a, m, d_b, k, &beta, d_c, m));
    CUDA_SAFE(cudaMemcpy(c2, d_c, m * n * sizeof(float),
                cudaMemcpyDeviceToHost));
    CUBLAS_SAFE(cublasDestroy(handle));

    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            if (rel_error(c[i * n + j], c2[i * n + j]) > 1e-6) {
                printf("c[%d, %d] = %f != %f\n",
                       i, j, c[i * n + j], c2[i * n + j]);
                goto loop_break;
            }
        }
    }
loop_break:
    free(c);
}
