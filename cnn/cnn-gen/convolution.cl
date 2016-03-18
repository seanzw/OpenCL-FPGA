#ifdef __xilinx__
__attribute__((reqd_work_group_size(WORK_GROUP_DIM_0, WORK_GROUP_DIM_1, WORK_GROUP_DIM_2)))
#endif
__kernel void KERNEL_NAME(
    KERNEL_PARAM
    __constant float *weight,
    __constant float *offset
    ) {

    int c = get_global_id(0);
    int r = get_global_id(1);
    int o = get_global_id(2);

    int cLocal = get_local_id(0);
    int rLocal = get_local_id(1);
    int oLocal = get_local_id(2);

    __local float inLocal[IWIDTH * IHEIGHT * IDEPTH];
    __local float weightLocal[IDEPTH * WORK_GROUP_DIM_2 * KERNEL_LEN];

    // This the the first work item in the group,
    // Copy the input and weight into the local buffer.
    if (cLocal == 0 && rLocal == 0 && oLocal == 0) {

            #ifdef __xilinx__
            __attribute__((xcl_pipeline_loop))
            #endif
            for (int i = 0; i < IWIDTH * IHEIGHT * IDEPTH; ++i) {
                inLocal[i] = in[i];
            }

            #ifdef __xilinx__
            __attribute__((xcl_pipeline_loop))
            #endif
            for (int i = 0; i < IDEPTH * WORK_GROUP_DIM_2 * KERNEL_LEN; ++i) {
                weightLocal[i] = weight[o * IDEPTH * KERNEL_LEN + i];
            }
    }

    // Set a barrier.
    barrier(CLK_LOCAL_MEM_FENCE);

    if (c < OWIDTH && r < OHEIGHT && o < ODEPTH) {

        float sum = 0.0f;


        // For each input feature map.
        #ifdef __xilinx__
        __attribute__((xcl_pipeline_loop))
        #endif
        for (int i = 0; i < IDEPTH; ++i) {

            float inputBuf[KERNEL_LEN];
            float weightBuf[KERNEL_LEN];
            int idx = 0;
            int weightBase = (oLocal * IDEPTH + i) * KERNEL_LEN;
                #ifdef __xilinx__
                __attribute__((xcl_pipeline_loop))
                #endif
                for (int x = 0; x < KERNEL_SIZE; ++x) {
                        #ifdef __xilinx__
                        __attribute__((xcl_pipeline_loop))
                        #endif
                        for (int y = 0; y < KERNEL_SIZE; ++y) {
                            inputBuf[idx] = inLocal[(i * IHEIGHT + r + x) * IWIDTH + c + y];
                            weightBuf[idx] = weightLocal[weightBase + idx];
                            idx++;
                        }
                }

                #ifdef __xilinx__
                __attribute__((xcl_pipeline_loop))
                #endif
                for (int x = 0; x < KERNEL_LEN; ++x) {
                    sum += inputBuf[x] * weightBuf[x];
                }
        }

        // Get the output index.
        int outIdx = (o * OHEIGHT + r) * OWIDTH + c;
        out[outIdx] = sigmod(sum + offset[o]);
    }
}