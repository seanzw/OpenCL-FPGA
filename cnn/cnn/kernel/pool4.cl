float sigmod(float in) {
    return 1.0f / (1.0f + exp(-in)); 
}
#define KERNEL_SIZE 2
#define KERNEL_LEN 4
#define IWIDTH 10
#define IHEIGHT 10
#define IDEPTH 16
#define IN_SIZE 1600
#define OWIDTH 5
#define OHEIGHT 5
#define ODEPTH 16
#define OWIDTH_TILE 1
#define OHEIGHT_TILE 1
#define ODEPTH_TILE 1
#define IDEPTH_TILE 1
#define OUT_SIZE 400
#define WORK_GROUP_DIM_0 16
#define WORK_GROUP_DIM_1 1
#define WORK_GROUP_DIM_2 1
#define KERNEL_NAME pool4
#define KERNEL_PARAM __global float *in, __global float *out,
__attribute__((reqd_work_group_size(WORK_GROUP_DIM_0, WORK_GROUP_DIM_1, WORK_GROUP_DIM_2)))
__kernel void KERNEL_NAME(
    KERNEL_PARAM
    __global float *weight,
    __global float *offset) {
    int c = get_global_id(0);
    int r = get_global_id(1);
    int o = get_global_id(2);

    int cLocal = get_local_id(0);
    int rLocal = get_local_id(1);
    int oLocal = get_local_id(2);

    __local float inLocal[IWIDTH * IHEIGHT * IDEPTH];
    __local float weightLocal[WORK_GROUP_DIM_2];
    __local float offsetLocal[WORK_GROUP_DIM_2];
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
            for (int i = 0; i < WORK_GROUP_DIM_2; ++i) {
                weightLocal[i] = weight[o + i];
                offsetLocal[i] = offset[o + i];
            }
    }

    // Set a barrier.
    barrier(CLK_LOCAL_MEM_FENCE);

    if (c < OWIDTH && r < OHEIGHT && o < ODEPTH) {

        float sum = 0.0f;

        for (int x = 0; x < KERNEL_SIZE; ++x) {
            for (int y = 0; y < KERNEL_SIZE; ++y) {
                sum += inLocal[(o * IHEIGHT + r * KERNEL_SIZE + x) * IWIDTH + c * KERNEL_SIZE + y];
            }
        }

        sum = sum * weightLocal[oLocal] + offsetLocal[oLocal];

        // Get the output index.
        int outIdx = (o * OHEIGHT + r) * OWIDTH + c;
        out[outIdx] = sigmod(sum);
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
#undef OWIDTH_TILE
#undef OHEIGHT_TILE
#undef ODEPTH_TILE
#undef IDEPTH_TILE
#undef OUT_SIZE
#undef WORK_GROUP_DIM_0
#undef WORK_GROUP_DIM_1
#undef WORK_GROUP_DIM_2
#undef KERNEL_NAME
#undef KERNEL_PARAM
