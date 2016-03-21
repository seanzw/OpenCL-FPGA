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

    // This the the first work item in the group,
    // Copy the input, output and weight into the local buffer.
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
        for (int i = 0; i < IDEPTH * ODEPTH * KERNEL_LEN; ++i) {
            weightLocal[i] = weight[i];
        }
    }

    // Set a barrier.
    barrier(CLK_LOCAL_MEM_FENCE);

    // Initialize the private output buffer to zero.
    float outPrivate[OWIDTH_TILE * OHEIGHT_TILE * ODEPTH_TILE];
    for (int i = 0; i < OWIDTH_TILE * OHEIGHT_TILE * ODEPTH_TILE; ++i) {
        outPrivate[i] = 0.0f;
    }

    // Tile the input feature map.
    for (int iTile = 0; iTile < IDEPTH; iTile += IDEPTH_TILE) {

        int oPrivateIdx = 0;
        for (int o = oTile; o < oTile + ODEPTH_TILE; ++o) {
            
            for (int r = rTile; r < rTile + OHEIGHT_TILE; ++r) {

                for (int c = cTile; c < cTile + OWIDTH_TILE; ++c, ++oPrivateIdx) {

                    for (int i = iTile; i < iTile + IDEPTH_TILE; ++i) {

                        int weightIdx = 0;
                        for (int x = 0; x < KERNEL_SIZE; ++x) {

                            for (int y = 0; y < KERNEL_SIZE; ++y, ++weightIdx) {

                                outPrivate[oPrivateIdx] += inLocal[(i * IHEIGHT + r + x) * IWIDTH + c + y]
                                    * weightLocal[(o * IDEPTH + i) * KERNEL_LEN + weightIdx];
                            }
                        }
                    }
                }
            }
        }
    }

    // Store the output buffer to global buffer.
    int oPrivateIdx = 0;
    for (int o = oTile; o < oTile + ODEPTH_TILE; ++o) {
        for (int r = rTile; r < rTile + OHEIGHT_TILE; ++r) {
            for (int c = cTile; c < cTile + OWIDTH_TILE; ++c, ++oPrivateIdx) {
                out[(o * OHEIGHT + r) * OWIDTH + c] = sigmod(outPrivate[oPrivateIdx] + offset[o]);
            }
        }
    }
}