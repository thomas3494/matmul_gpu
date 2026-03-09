/* Wellford's algorithm for mean and stddev */
#define BENCH(statement, gflop, iter)                                          \
{                                                                              \
    for (int t = 0; t < 3; t++) {                                              \
        statement;                                                             \
    }                                                                          \
    double mean = 0.0;                                                         \
    double M2   = 0.0;                                                         \
    float duration;                                                            \
    cudaEvent_t start, stop;                                                   \
    cudaEventCreate(&start);                                                   \
    cudaEventCreate(&stop);                                                    \
    for (int t = 0; t < iter; t++) {                                           \
        cudaEventRecord(start, 0);                                             \
        statement;                                                             \
        cudaEventRecord(stop, 0);                                              \
        cudaEventSynchronize(stop);                                            \
        cudaEventElapsedTime(&duration, start,stop);                           \
        float gflops = 1e3 * gflop / duration;                                 \
        double delta = gflops - mean;                                          \
        mean += delta / (t + 1);                                               \
        double delta2 = gflops - mean;                                         \
        M2 += delta * delta2;                                                  \
    }                                                                          \
    printf("%lf,%lf\n", mean, sqrt(M2 / iter));                                \
}
