float sigmod(float in) {
    return 1.0f / (1.0f + exp(-in)); 
}
#define KERNEL_SIZE 5
#define KERNEL_LEN 25
#define IWIDTH 32
#define IHEIGHT 32
#define IDEPTH 1
#define OWIDTH 28
#define OHEIGHT 28
#define ODEPTH 6
#define WORK_GROUP_DIM_0 28
#define WORK_GROUP_DIM_1 28
#define WORK_GROUP_DIM_2 3
#define KERNEL_NAME conv1
#define KERNEL_PARAM __global float *in, __global float *out,
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

        for (int i = 0; i < IWIDTH * IHEIGHT * IDEPTH; ++i) {
            inLocal[i] = in[i];
        }

        for (int i = 0; i < IDEPTH * WORK_GROUP_DIM_2 * KERNEL_LEN; ++i) {
            weightLocal[i] = weight[o * IDEPTH * KERNEL_LEN + i];
        }
    }

    // Set a barrier.
    barrier(CLK_LOCAL_MEM_FENCE);

    if (c < OWIDTH && r < OHEIGHT && o < ODEPTH) {

        float sum = 0.0f;
        int weightBase = (oLocal * IDEPTH) * KERNEL_LEN;

        // For each input feature map.
        for (int i = 0; i < IDEPTH; ++i) {

            float inputBuf[KERNEL_LEN];
            float weightBuf[KERNEL_LEN];
            int idx = 0;
            for (int x = 0; x < KERNEL_SIZE; ++x) {
                for (int y = 0; y < KERNEL_SIZE; ++y) {
                    inputBuf[idx] = inLocal[(i * IHEIGHT + r + x) * IWIDTH + c + y];
                    weightBuf[idx] = weightLocal[weightBase + idx];
                    idx++;
                }
            }

            for (int x = 0; x < KERNEL_LEN; ++x) {
                sum += inputBuf[x] * weightBuf[x];
            }
            weightBase += KERNEL_LEN;
        }

        // Get the output index.
        int outIdx = (o * OHEIGHT + r) * OWIDTH + c;
        out[outIdx] = sigmod(sum + offset[o]);
    }
}
#undef KERNEL_SIZE
#undef KERNEL_LEN
#undef IWIDTH
#undef IHEIGHT
#undef IDEPTH
#undef OWIDTH
#undef OHEIGHT
#undef ODEPTH
#undef WORK_GROUP_DIM_0
#undef WORK_GROUP_DIM_1
#undef WORK_GROUP_DIM_2
#undef KERNEL_NAME
#undef KERNEL_PARAM
