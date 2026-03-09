#include <stdio.h>
#include <stdlib.h>
#include <cublas_v2.h>
#include <debug.h>
#include <bench.h>
#include <test_code.h>

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

    cublasHandle_t handle;
    CUBLAS_SAFE(cublasCreate(&handle));
    const float alpha = 1.0;
    const float beta = 0.0;

#ifndef PROFILE
    BENCH(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k,
                      &alpha, d_a, m, d_b, k, &beta, d_c, m),
                      1e-9 * 2.0 * m * n * k, 10);
    check(d_c, d_a, d_b, m, k, n);
#else
    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k,
                &alpha, d_a, m, d_b, k, &beta, d_c, m);
#endif

    CUBLAS_SAFE(cublasDestroy(handle));
    CUDA_SAFE(cudaFree(d_a));
    CUDA_SAFE(cudaFree(d_b));
    CUDA_SAFE(cudaFree(d_c));

    return EXIT_SUCCESS;
}
