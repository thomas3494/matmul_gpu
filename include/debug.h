#pragma once

#include <cublas_v2.h>

#define CUDA_SAFE(fncall)                                                     \
    {                                                                         \
        cudaError_t err = fncall;                                             \
        if (err != cudaSuccess) {                                             \
            printf("%s:%d %s\n", __FILE__, __LINE__,                          \
                    cudaGetErrorString(err));                                 \
        }                                                                     \
    }

#define CUBLAS_SAFE(fncall)                                                   \
    {                                                                         \
        cublasStatus_t err = fncall;                                          \
        if (err != CUBLAS_STATUS_SUCCESS) printf("%s\n",                      \
               cublasGetStatusString(err));                                   \
    }

#define TIME(duration, fncalls)                                               \
    {                                                                         \
        struct timeval tv1, tv2;                                              \
        gettimeofday(&tv1, NULL);                                             \
        fncalls                                                               \
        gettimeofday(&tv2, NULL);                                             \
        duration = (double) (tv2.tv_usec - tv1.tv_usec) / 1000000 +           \
         (double) (tv2.tv_sec - tv1.tv_sec);                                  \
    }
