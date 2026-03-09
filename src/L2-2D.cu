/**
 * We use L2-cache so we can block over all SM, increasing reuse.
 * Important to note: L2 is not that much faster than VRAM.
 *
 * Suppose we have 4 SM's. Then we would compute the block marked by x's as
 * follows:
 *
 *              (1|1     )
 *              (2|2     )
 *              (3|3     )
 *              (4|4     )
 *
 *  (1|2|3|4)   (x|x|    )
 *  (1|2|3|4)   (x|x|    )
 *  (       )   (        )
 *  (       )   (        )
 *
 * Here 1, 2, 3, 4, represent the order in which we update the block.
 * Both A and B are reused by two SMs now.
 *
 * To hide latency, we should block the k-loop and have each SM in a row/column
 * start at a different point.
 *
 * In general, our processor grid will not fit the block grid.
 * 
 *   <-> gridDim.x
 *  ^
 *  | gridDim.y
 *  v
 *
 *  23 x 2
 *
 *  
 *  (x x        )
 *  (x x        )
 *  (x x        )
 *  (           )
 *  (           )
 *  (    x x    )
 *  (           )
 *  (           )
 *  (x x        )
 *  (x x        )
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
void bcast_row(float b_reg[NR], vec4 b_sh[KB][NB / 4], int p)
{
    int t2 = threadIdx.x % (NB / NR);

    vec4 *res = (vec4 *)b_reg;
    for (int i = 0; i < NR / 4; i++) {
        res[i] = b_sh[p][t2 * NR / 4];
    }
}

template<int MB, int KB, int THREADS>
__device__ void load_A(float a_sh[MB][KB], vec4 *a, int ld)
{
    /**
     * Thread space: [4 * THREADS / KB, KB / 4]
     **/
    int t1 = threadIdx.x / (KB / 4);
    int t2 = threadIdx.x % (KB / 4);

    for (; t1 < MB; t1 += 4 * THREADS / KB) {
        vec4 *sh_vec = (vec4 *)(&a_sh[t1][4 * t2]);
        sh_vec[0] = a[t1 * ld];
    }
}

template<int KB, int NB, int THREADS>
__device__ void load_B(vec4 b_sh[KB][NB / 4], const vec4 *b, int ld)
{
    /**
     * Thread space: [4 * blockDim.x / NB, NB / 4]
     **/
    int t1 = threadIdx.x / (NB / 4);
    int t2 = threadIdx.x % (NB / 4);
    for (; t1 < KB; t1 += 4 * THREADS / NB) {
        b_sh[t1][t2] = b[t1 * ld];
    }
}

template<int MB, int KB, int NB, int THREADS>
__device__
void load_pp(float a_sh[MB][KB], vec4 b_sh[KB][NB / 4],
             vec4 *a, vec4 *b, int k, int n, int pp)
{
    load_A<MB, KB, THREADS>(a_sh, a + pp / 4, k / 4);
    load_B<KB, NB, THREADS>(b_sh, b + pp * n / 4, n / 4);
}

template<int MB, int KB, int NB, int MR, int NR, int THREADS>
__device__
void comp_pp(float c_reg[MR][NR], float a_reg[MR], float b_reg[NR],
             float a_sh[MB][KB], vec4 b_sh[KB][NB / 4])
{
    for (int p = 0; p < KB; p++) {
        bcast_col<MB, KB, NB, MR, NR>(b_reg, a_sh, p);
        bcast_row<KB, NB, NR>(b_reg, b_sh, p);
        outer_product<MR, NR>(c_reg, a_reg, b_reg);
    }
}

/**
 * Sums outer product of 
 *    a[k_block:k_block + KB * gridDim.y - 1 .] and 
 *    and
 *    b[., k_block:k_block + KB * gridDim.y]
 * Blocks start at a different offset to collectively read this into L2
 * as fast as possible.
 **/
template<int MB, int KB, int NB, int MR, int NR, int THREADS>
__device__
void block_iter(vec4 *my_a, vec4 *my_b, int m, int k, int n,
                float a_sh[MB][KB], vec4 b_sh[KB][NB / 4],
                float c_reg[MR][NR], float a_reg[MR], float b_reg[NR],
                int k_block, int block_id)
{
    int pp = k_block + block_id * KB;
    for (int i = 0; i < gridDim.y; i++) {
        if (pp < k) {
            load_pp<MB, KB, NB, THREADS>(a_sh, b_sh, my_a, my_b, k, n, pp);
            __syncthreads();
            /* Profiling shows we are now waiting at a barrier, which
             * is why this is not actually faster, despite moving less
             * memory. */
            comp_pp<MB, KB, NB, MR, NR, THREADS>
                   (c_reg, a_reg, b_reg, a_sh, b_sh);
        }
        pp += KB;
        if (pp == k_block + KB * gridDim.y) {
            pp = k_block;
        }
    }
}

template<int MB, int KB, int NB, int MR, int NR, int THREADS>
__global__
void matmul_kernel(float *c, float *a, float *b, int m, int k, int n)
{
    assert(k % KB == 0);
    assert(m >= gridDim.y * MB);

    __shared__ float a_sh[MB][KB];
    __shared__ vec4  b_sh[KB][NB / 4];

    float c_reg[MR][NR] = {0.f};
    float a_reg[MR];
    float b_reg[NR];

    int t1       = threadIdx.x / (NB / NR);
    int t2       = threadIdx.x % (NB / NR);
    int blocks_x = (m / MB);
    int blocks_y = (n / NB);
    int block    = blockIdx.y * gridDim.x + blockIdx.x;

    int b1 = blockIdx.y;
    int b2 = blockIdx.x;

    for (; block < blocks_x * blocks_y; block += gridDim.x * gridDim.y) {
        /**
         * Globally, we have
         *    Thread space: [MB / MR, NB / NR]
         *    Block space:  [m  / MB, n  / NB]
         **/
        vec4 *my_a = (vec4 *)(a + b1 * MB * k + 4 * threadIdx.x % (KB / 4));
        vec4 *my_b = (vec4 *)(b + b2 * NB     + 4 * threadIdx.x % (NB / 4));
        vec4 *my_c = (vec4 *)(c + b1 * MB * n + b2 * NB + t2 * NR);

        /**
         * Collectively, the SMs read in KB * gridDim.y rows.
         **/
        for (int k_block = 0; k_block < k; k_block += KB * gridDim.y)
        {
            block_iter<MB, KB, NB, MR, NR, THREADS>
                (my_a, my_b, m, k, n,
                 a_sh, b_sh, c_reg, a_reg, b_reg, k_block, blockIdx.y);
        }

        for (int i = 0; i < MR; i++) {
            int glob_i = t1 * MR + i;
            vec4 *c_reg_vec = (vec4 *)(c_reg[i]);
            for (int j = 0; j < NR / 4; j++) {
                my_c[glob_i * n / 4 + j] = c_reg_vec[j];
                c_reg_vec[j] = make_float4(0.f, 0.f, 0.f, 0.f);
            }
        }

        b1 += gridDim.y;
        if (b1 >= blocks_x) {
            b1 -= blocks_x;
            b2 += gridDim.x;
        }
    }
}

/**
 * Factorizes n into n1 * n2, with n1 >= n2
 **/
void factor(int n /* in */, int *n1 /* out */, int *n2 /* out */)
{
    int divisor = floor(sqrt(n));
    while (n % divisor != 0) {
        divisor--;
    }
    *n2 = divisor;
    *n1 = n / *n2;
}

/**
 * MB: reuse of B from L2      -> shared
 * KB: allows for vectorized loads of B, some latency hiding,
       and instruction level parallelism for loads
 * NB: reuse of A from L2     -> shared
 * MR: reuse of B from shared -> registers
 * NR: reuse of A from shared -> registers
 **/
template<int MB = 128, int KB = 32, int NB = 256, int MR = 8, int NR = 8>
void matmul(float *c, float *a, float *b, int m, int k, int n)
{
    constexpr int THREADS = (MB / MR) * (NB / NR);

    cudaDeviceProp deviceProp;
    CUDA_SAFE(cudaGetDeviceProperties(&deviceProp, 0));
    int blocks_x, blocks_y;
    /* Could be very uneven, we want blocks_y * MB approx blocks_x * NB.
     * Is there a better way that is not too hard with load-balancing? */
    factor(deviceProp.multiProcessorCount, &blocks_y, &blocks_x);
    dim3 blocks (blocks_x, blocks_y);
    dim3 threads(THREADS);

    matmul_kernel<MB, KB, NB, MR, NR, THREADS>
                 <<<blocks, threads>>>
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
