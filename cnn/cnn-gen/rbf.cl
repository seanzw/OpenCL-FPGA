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
