/**
 * The idea is as follows.
 * To compute the column of C marked with x's, we read that column
 * of B in order 1, 2, 3, 4. So if these columns are computed at the same
 * time, we can store it in L2 (4 MB on 3070).
 * 
 *
 *              (1|      )
 *              (2|      )
 *              (3|      )
 *              (4|      )
 *
 *  (1|2|3|4)   (x|      )
 *  (1|2|3|4)   (x|      )
 *  (1|2|3|4)   (x|      )
 *  (1|2|3|4)   (x|      )
 *
 * We make the blocks rectangular NB > MB, so we have enough blocks per column
 * to satisfy the SMs. Each thread block then loops over the columns.
 * The reuse of L2 is multiplied by the number of SMs, so small MB is ok.
 * 32 x 1024.
 *
 * The problem with this approach is that each SM reads the same element
 * at the same time, so it does not help with latency.
 *
 * Should block over k as well, and have each SM start reading at a different
 * offset. Then for the first iteration we have large latency, but after we
 * hit L2.
 *
 * But multiple thread blocks read at the same time, so doesn't really
 * help with latency. Maybe different blocks start reading at different points?
 **/

#include <stdio.h>
#include <stdlib.h>
#include <cublas_v2.h>
#include <debug.h>
#include <test_code.h>
#include <assert.h>

typedef float4 vec4;

template<int MR, int NR>
__device__
void outer_product(float c[MR][NR], float a[MR], float b[NR])
{
    /* We could use one register instead of MR per load of a,
     * but it does not take that many registers, so maybe not necessary. */
    for (int i = 0; i < MR; i++) {
        for (int j = 0; j < NR; j++) {
            c[i][j] += a[i] * b[j];
        }
    }
}

/**
 * Duplicates column p across all rows in a 2D thread grid.
 **/
template<int MB, int KB, int NB, int MR, int NR>
__device__
void bcast_col(float a_reg[MR], float a_sh[MB][KB], int p)
{
    int t1 = threadIdx.x / (NB / NR);

    for (int i = 0; i < MR; i++) {
        a_reg[i] = a_sh[t1 * MR + i][p];
    }
}

/**
 * Duplicates row p across all columns in a 2D thread grid.
 **/
template<int KB, int NB, int NR>
__device__
void bcast_row(float b_reg[NR], float b_sh[KB][NB], int p)
{
    int t2 = threadIdx.x % (NB / NR);

    vec4 *x   = (vec4 *)(&b_sh[p][t2 * NR]);
    vec4 *res = (vec4 *)b_reg;
    for (int i = 0; i < NR / 4; i++) {
        res[i] = x[i];
    }
}

template<int MB, int KB, int THREADS>
__device__ void load_A(float a_sh[MB][KB], float *a, int ld)
{
    /**
     * Thread space: [4 * THREADS / KB, KB / 4]
     **/
    int t1 = threadIdx.x / (KB / 4);
    int t2 = threadIdx.x % (KB / 4);

    /* Could make blockDim.x a macro as well. For this case, we know
     * the loop is precisely one iteration. */
    for (; t1 < MB; t1 += 4 * THREADS / KB) {
        vec4 *a_vec  = (vec4 *)(a + t1 * ld + 4 * t2);
        vec4 *sh_vec = (vec4 *)(&a_sh[t1][4 * t2]);
        sh_vec[0] = a_vec[0];
    }
}

template<int KB, int NB, int THREADS>
__device__ void load_B(float b_sh[KB][NB], const float *b, int ld)
{
    /**
     * Thread space: [4 * blockDim.x / NB, NB / 4]
     **/
    int t1 = threadIdx.x / (NB / 4);
    int t2 = threadIdx.x % (NB / 4);
    for (; t1 < KB; t1 += 4 * THREADS / NB) {
        vec4 *b_vec  = (vec4 *)(b + t1 * ld + 4 * t2);
        vec4 *sh_vec = (vec4 *)(&b_sh[t1][4 * t2]);
        sh_vec[0] = b_vec[0];
    }
}

template<int MB, int KB, int NB, int THREADS>
__device__
void load_pp(float a_sh[MB][KB], float b_sh[KB][NB],
             float *a, float *b, int k, int n, int pp)
{
    load_A<MB, KB, THREADS>(a_sh, a + pp    , k);
    load_B<KB, NB, THREADS>(b_sh, b + pp * n, n);
}

template<int MB, int KB, int NB, int MR, int NR, int THREADS>
__device__
void comp_pp(float c_reg[MR][NR], float a_reg[MR], float b_reg[NR],
             float a_sh[MB][KB], float b_sh[KB][NB])
{
    for (int p = 0; p < KB; p++) {
        bcast_col<MB, KB, NB, MR, NR>(b_reg, a_sh, p);
        bcast_row<KB, NB, NR>(b_reg, b_sh, p);
        outer_product<MR, NR>(c_reg, a_reg, b_reg);
    }
}

template<int MB, int KB, int NB, int MR, int NR, int THREADS>
__global__
void matmul_kernel(float *c, float *a, float *b, int m, int k, int n)
{
    assert(k % KB == 0);

    __shared__ float a_sh[MB][KB];
    __shared__ float b_sh[KB][NB];

    float c_reg[MR][NR] = {0.f};
    float a_reg[MR];
    float b_reg[NR];

    int t1       = threadIdx.x / (NB / NR);
    int t2       = threadIdx.x % (NB / NR);
    int blocks_x = (m / MB);
    int blocks_y = (n / NB);
    int block    = blockIdx.x;

    for (; block < blocks_x * blocks_y; block += gridDim.x) {
        int b1       = block % blocks_x;
        int b2       = block / blocks_x;

        /**
         * Thread space: [MB / MR, NB / NR]
         * Block space:  [m  / MB, n  / NB]
         **/
        float *my_a = a + b1 * MB * k;
        float *my_b = b + b2 * NB;
        float *my_c = c + b1 * MB * n + b2 * NB;

        /**
         * Collectively, the SMs read in KB * gridDim.x rows.
         **/
        for (int k_block = 0; k_block < k; k_block += KB * gridDim.x) 
        {
            /* We start at different offsets so we don't all request the same
             * data. */
            int pp = k_block + blockIdx.x * KB;
            for (int i = 0; i < gridDim.x; i++) {
                if (pp < k) {
                    load_pp<MB, KB, NB, THREADS>(a_sh, b_sh, my_a, my_b, k, n, pp);
                    __syncthreads();
                    comp_pp<MB, KB, NB, MR, NR, THREADS>
                           (c_reg, a_reg, b_reg, a_sh, b_sh);
                }
                pp += KB;
                if (pp == k_block + KB * gridDim.x) {
                    pp = k_block;
                }
            }
        }

        for (int i = 0; i < MR; i++) {
            for (int j = 0; j < NR; j++) {
                int glob_i = t1 * MR + i;
                int glob_j = t2 * NR + j;
                my_c[glob_i * n + glob_j] = c_reg[i][j];
                c_reg[i][j] = 0;
            }
        }
    }
}

/**
 * MB: reuse of B from global -> shared
 * KB: allows for vectorized loads of B, some latency hiding,
       and instruction level parallelism for loads
 * NB: reuse of A from global -> shared
 * MR: reuse of B from shared -> registers
 * NR: reuse of A from shared -> registers
 **/
template<int MB = 128, int KB = 8, int NB = 256, int MR = 8, int NR = 8>
void matmul(float *c, float *a, float *b, int m, int k, int n)
{
    constexpr int THREADS = (MB / MR) * (NB / NR);

    cudaDeviceProp deviceProp;
    CUDA_SAFE(cudaGetDeviceProperties(&deviceProp, 0));
    int blocks = deviceProp.multiProcessorCount;

    matmul_kernel<MB, KB, NB, MR, NR, THREADS>
                 <<<blocks, THREADS>>>
                 (c, a, b, m, k, n);
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

    fprintf(stderr, "Duration: %lf s\n", duration);
    fprintf(stderr, "Gflops/s: ");
    printf("%f\n", 2.0 * m * n * k / duration / 1e9);

    check(d_c, m, k, n);

    CUDA_SAFE(cudaFree(d_a));
    CUDA_SAFE(cudaFree(d_b));
    CUDA_SAFE(cudaFree(d_c));

    return EXIT_SUCCESS;
}
