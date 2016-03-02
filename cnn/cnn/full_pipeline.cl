float sigmod(float in) {
    return 1.0f / (1.0f + exp(-in)); 
}
#define IN_SIZE 400
#define OUT_SIZE 120
#ifdef __xilinx__
__attribute__((reqd_work_group_size(16, 1, 1)))
#endif
__kernel void full1(
    __global float *in,
    __global float *out,
    __global float *weight,
    __global float *offset
    ) {
#ifdef __xilinx__
    __attribute__((xcl_pipeline_workitems))
#endif
    int o = get_global_id(0);
    if (o < OUT_SIZE) {
        float sum = 0;
#ifdef __xilinx__
        __attribute__((xcl_pipeline_loop))
#endif
        for (int i = 0; i < IN_SIZE; i++) {
            sum += weight[o * IN_SIZE + i] * in[i];
        }
        sum += offset[o];
        out[o] = sigmod(sum);
    }
}
#undef IN_SIZE
#undef OUT_SIZE
