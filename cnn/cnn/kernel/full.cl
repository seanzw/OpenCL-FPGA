float sigmod(float in) {
    return 1.0f / (1.0f + exp(-in)); 
}
#define IN_SIZE 120
#define OUT_SIZE 84
#define BUF_SIZE 10
#ifdef __xilinx__
__attribute__((reqd_work_group_size(16, 1, 1)))
#endif
__kernel void full6(
    __global float *in,
    __global float *out,
    __global float *weight,
    __global float *offset
    ) {
    int o = get_global_id(0);

    float inBuf[BUF_SIZE];
    float weightBuf[BUF_SIZE];

    if (o < OUT_SIZE) {
        float sum = 0;
        #ifdef __xilinx__
                __attribute__((xcl_pipeline_loop))
        #endif
        for (int i = 0; i < IN_SIZE; i += BUF_SIZE) {

            #ifdef __xilinx__
            __attribute__((opencl_unroll_hint))
            #endif
            for (int j = 0; j < BUF_SIZE; ++j) {
                inBuf[j] = in[i + j];
                weightBuf[j] = weight[o * IN_SIZE + i + j];
            }

            #ifdef __xilinx__
            __attribute__((opencl_unroll_hint))
            #endif
            for (int j = 0; j < BUF_SIZE; ++j) {
                sum += weightBuf[j] * inBuf[j];
            }
        }
        sum += offset[o]; 
        out[o] = sigmod(sum);
    }
}
#undef IN_SIZE
#undef OUT_SIZE
#undef BUF_SIZE
