float sigmod(float in) {
    return 1.0f / (1.0f + exp(-in)); 
}
#define KERNEL_SIZE 14
#define KERNEL_LEN 196
#define IWIDTH 84
#define IHEIGHT 1
#define IDEPTH 1
#define IN_SIZE 84
#define OWIDTH 10
#define OHEIGHT 1
#define ODEPTH 1
#define OUT_SIZE 10
#define WORK_GROUP_DIM_0 10
#define WORK_GROUP_DIM_1 1
#define WORK_GROUP_DIM_2 1
#define KERNEL_NAME rbf7
#define KERNEL_PARAM __global float *in, __global float *out,
__attribute__((reqd_work_group_size(WORK_GROUP_DIM_0, WORK_GROUP_DIM_1, WORK_GROUP_DIM_2)))
__kernel void KERNEL_NAME(
    KERNEL_PARAM
    __global float *weight,
    __global float *offset
    ) {

    int o = get_global_id(0);
    int oLocal = get_local_id(0);

    __local float inLocal[IN_SIZE];
    __local float weightLocal[IN_SIZE * WORK_GROUP_DIM_0];

    if (oLocal == 0) {
        for (int i = 0; i < IN_SIZE; ++i) {
            inLocal[i] = in[i];
        }
        for (int i = 0; i < IN_SIZE * WORK_GROUP_DIM_0; ++i) {
            weightLocal[i] = weight[o * IN_SIZE + i];
        }
    }

    // Set a barrier.
    barrier(CLK_LOCAL_MEM_FENCE);
    
    if (o < OUT_SIZE) {
        float sum = 0.0f;

        float inBuf[KERNEL_SIZE];
        float weightBuf[KERNEL_SIZE];

        #ifdef __xilinx__
        __attribute__((xcl_pipeline_loop))
        #endif
        for (int i = 0; i < IN_SIZE; i += KERNEL_SIZE) {
        
            #ifdef __xilinx__
            __attribute__((opencl_unroll_hint))
            #endif
            for (int j = 0; j < KERNEL_SIZE; ++j) {
                inBuf[j] = inLocal[i + j];
                weightBuf[j] = weightLocal[oLocal * IN_SIZE + i + j];
            }
        
            #ifdef __xilinx__
            __attribute__((opencl_unroll_hint))
            #endif
            for (int j = 0; j < KERNEL_SIZE; ++j) {
                float diff = weightBuf[j] - inBuf[j];
                sum += diff * diff;
            }
        }
        out[o] = sum;
    }
}

#undef KERNEL_SIZE
#undef KERNEL_LEN
#undef IWIDTH
#undef IHEIGHT
#undef IDEPTH
#undef IN_SIZE
#undef OWIDTH
#undef OHEIGHT
#undef ODEPTH
#undef OUT_SIZE
#undef WORK_GROUP_DIM_0
#undef WORK_GROUP_DIM_1
#undef WORK_GROUP_DIM_2
#undef KERNEL_NAME
#undef KERNEL_PARAM
