float sigmod(float in) {
    return 1.0f / (1.0f + exp(-in)); 
}
#define KERNEL_SIZE 10
#define KERNEL_LEN 100
#define IWIDTH 1
#define IHEIGHT 1
#define IDEPTH 120
#define IN_SIZE 120
#define OWIDTH 84
#define OHEIGHT 1
#define ODEPTH 1
#define OUT_SIZE 84
#define WORK_GROUP_DIM_0 12
#define WORK_GROUP_DIM_1 1
#define WORK_GROUP_DIM_2 1
#define KERNEL_NAME full6
#define KERNEL_PARAM __global float *in, __global float *out,
__attribute__((reqd_work_group_size(WORK_GROUP_DIM_0, 1, 1)))
__kernel void KERNEL_NAME(
    KERNEL_PARAM
    __constant float *weight,
    __constant float *offset
    ) {

    int o = get_global_id(0);
    int oLocal = get_local_id(0);

    __local float inLocal[IN_SIZE];
    __local float weightLocal[WORK_GROUP_DIM_0 * IN_SIZE];
    __local float offsetLocal[WORK_GROUP_DIM_0];

    if (oLocal == 0) {

        for (int i = 0; i < IN_SIZE; ++i) {
            inLocal[i] = in[i];
        }

        for (int i = 0; i < WORK_GROUP_DIM_0 * IN_SIZE; ++i) {
            weightLocal[i] = weight[o * IN_SIZE + i];
        }

        for (int i = 0; i < WORK_GROUP_DIM_0; ++i) {
            offsetLocal[i] = offset[o + i];
        }
    }

    // Set a barrier.
    barrier(CLK_LOCAL_MEM_FENCE);

    if (o < OUT_SIZE) {

        float sum = 0;
        #ifdef __xilinx__
                __attribute__((xcl_pipeline_loop))
        #endif
        float inBuf[KERNEL_SIZE];
        float weightBuf[KERNEL_SIZE];
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
                sum += weightBuf[j] * inBuf[j];
            }
        }
        sum += offsetLocal[oLocal]; 
        out[o] = sigmod(sum);
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
