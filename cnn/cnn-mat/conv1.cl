float sigmod(float in) {
    return 1.0f / (1.0f + exp(-in));
}

#define M 6
#define N 25
#define P 784
#define N_UNROLL 5

__attribute__((reqd_work_group_size(1, 1, 1)))
__kernel void conv1(__global float* A, __global float* B, __global float* C) {

    for (int m = 0; m < M; ++m) {
        for (int p = 0; p < P; ++p) {
            float sum = 0.0f;
            for (int n = 0; n < N; n += N_UNROLL) {
                for (int i = 0; i < N_UNROLL; ++i) {
                    sum += A[m * N + n + i] * B[(n + i) * P + p];
                }
            }
            C[m * P + p] = sum;
        }
    }
}