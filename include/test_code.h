#pragma once

#include <cuda.h>
#include <debug.h>

__host__ __device__
int ceil(int a, int b)
{
    return (a + b - 1) / b;
}

struct InitA {
    __device__ __host__ float operator()(int i, int j) const {
        return float(i) + 1;
    }
};

struct InitB {
    __device__ __host__ float operator()(int i, int j) const {
        return float(j) + 2;
    }
};

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

    init(*d_a, m, k, InitA());
    init(*d_b, k, n, InitB());
    init(*d_c, m, n, Zero());
}

float rel_error(float a, float b)
{
    return fabs(a - b) / b;
}

void check(float *d_c, size_t m, size_t k, size_t n)
{
    float *c = (float *)malloc(m * n * sizeof(float));
    CUDA_SAFE(cudaMemcpy(c, d_c, m * n * sizeof(float),
                cudaMemcpyDeviceToHost));

    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            /* We can incur 2^(-24) per addition... */
            if (rel_error(c[i * n + j], k * (i + 1) * (j + 2)) > k * 1e-7) {
                printf("c[%d, %d] = %f\n", i, j, c[i * n + j]);
                goto loop_break;
            }
        }
    }
loop_break:
    free(c);
}
