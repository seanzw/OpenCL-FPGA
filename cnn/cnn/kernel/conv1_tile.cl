float sigmod(float in) {
    return 1.0f / (1.0f + exp(-in)); 
}
#define KERNEL_SIZE 5
#define KERNEL_LEN 25
#define IWIDTH 32
#define IHEIGHT 32
#define IDEPTH 1
#define IN_SIZE 1024
#define OWIDTH 28
#define OHEIGHT 28
#define ODEPTH 6
#define OWIDTH_TILE 4
#define OHEIGHT_TILE 4
#define ODEPTH_TILE 3
#define IDEPTH_TILE 1
#define OUT_SIZE 4704
#define WORK_GROUP_DIM_0 7
#define WORK_GROUP_DIM_1 7
#define WORK_GROUP_DIM_2 2
#define KERNEL_NAME conv1
#define KERNEL_PARAM __global float *in, __global float *out,
__attribute__((reqd_work_group_size(WORK_GROUP_DIM_0, WORK_GROUP_DIM_1, WORK_GROUP_DIM_2)))
__kernel void KERNEL_NAME(
    KERNEL_PARAM
    __constant float *weight,
    __constant float *offset
    ) {

    int cTile = get_global_id(0) * OWIDTH_TILE;
    int rTile = get_global_id(1) * OHEIGHT_TILE;
    int oTile = get_global_id(2) * ODEPTH_TILE;

    int cLocal = get_local_id(0);
    int rLocal = get_local_id(1);
    int oLocal = get_local_id(2);

    __local float inLocal[IN_SIZE];
    __local float weightLocal[IDEPTH * ODEPTH * KERNEL_LEN];
    __local float offsetLocal[ODEPTH];
    // __local float outLocal[OUT_SIZE];

    // This the the first work item in the group,
    // Copy the input, output and weight into the local buffer.
    if (cLocal == 0 && rLocal == 0 && oLocal == 0) {

        #ifdef __xilinx__
        __attribute__((xcl_pipeline_loop))
        #endif
        for (int i = 0; i < IN_SIZE; ++i) {
            inLocal[i] = in[i];
        }

        #ifdef __xilinx__
        __attribute__((xcl_pipeline_loop))
        #endif
        for (int i = 0; i < IDEPTH * ODEPTH * KERNEL_LEN; ++i) {
            weightLocal[i] = weight[i];
        }


        #ifdef __xilinx__
        __attribute__((xcl_pipeline_loop))
        #endif
        for (int i = 0; i < ODEPTH; ++i) {
            offsetLocal[i] = offset[i];
        }
    }

    // Set a barrier.
    barrier(CLK_LOCAL_MEM_FENCE);

    // Initialize the private output buffer to zero.
    float outPrivate[OWIDTH_TILE * OHEIGHT_TILE * ODEPTH_TILE];
    #ifdef __xilinx__
    __attribute__((xcl_pipeline_loop))
    #endif
    for (int i = 0; i < OWIDTH_TILE * OHEIGHT_TILE * ODEPTH_TILE; ++i) {
        outPrivate[i] = 0.0f;
    }

    // Tile the input feature map.
    for (int iTile = 0; iTile < IDEPTH; iTile += IDEPTH_TILE) {

        int oPrivateIdx = 0;
        for (int r = 0; r < OHEIGHT_TILE; ++r) {
            for (int c = 0; c < OWIDTH_TILE; ++c) {
                for (int o = 0; o < ODEPTH_TILE; ++o, ++oPrivateIdx) {
                    for (int i = 0; i < IDEPTH_TILE; ++i) {
                        int weightIdx = 0;
                        for (int x = 0; x < KERNEL_SIZE; ++x) {
                            for (int y = 0; y < KERNEL_SIZE; ++y, ++weightIdx) {
                                outPrivate[oPrivateIdx] += inLocal[((i + iTile) * IHEIGHT + r + rTile + x) * IWIDTH + c + cTile + y]
                                    * weightLocal[((o + oTile) * IDEPTH + i + iTile) * KERNEL_LEN + weightIdx];
                            }
                        }
                    }
                }
            }
        }
    }

    // Store the output buffer to local buffer.
    int oPrivateIdx = 0;
    for (int r = 0; r < OHEIGHT_TILE; ++r) {
        for (int c = 0; c < OWIDTH_TILE; ++c) {
            for (int o = 0; o < ODEPTH_TILE; ++o, ++oPrivateIdx) {
                out[(o * OHEIGHT + r + rTile) * OWIDTH + c + cTile] = sigmod(outPrivate[oPrivateIdx] + offsetLocal[o + oTile]);
            }
        }
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
