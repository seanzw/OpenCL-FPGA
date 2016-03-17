float sigmod(float in) {
    return 1.0f / (1.0f + exp(-in)); 
}
__global float buf1[4704];
__global float buf2[1176];
__global float buf3[1600];
__global float buf4[400];
__global float buf5[120];
__global float buf6[84];
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
#define out buf1
#define KERNEL_PARAM __global float *in, 
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
#undef out
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
#define KERNEL_SIZE 2
#define IWIDTH 28
#define IHEIGHT 28
#define IDEPTH 6
#define OWIDTH 14
#define OHEIGHT 14
#define ODEPTH 6
#define in buf1
#define out buf2
#ifdef __xilinx__
__attribute__ ((reqd_work_group_size(16, 1, 1)))
#endif
__kernel void max2(
    __global float *weight,
    __global float *offset) {
    int c = get_global_id(0);
    int r = get_global_id(1);

    if (c < OWIDTH && r < ODEPTH * OHEIGHT) {

        // Get the index of the element in output feature map.
        int o = r / OHEIGHT;
        r = r % OHEIGHT;

        float sum = 0.0f;

        for (int x = 0; x < KERNEL_SIZE; ++x) {
            for (int y = 0; y < KERNEL_SIZE; ++y) {
                sum += in[(o * IHEIGHT + r * KERNEL_SIZE + x) * IWIDTH + c * KERNEL_SIZE + y];
            }
        }

        sum = sum * weight[o] + offset[o];

        // Get the output index.
        int outIdx = (o * OHEIGHT + r) * OWIDTH + c;
        out[outIdx] = sigmod(sum);
    }
}

#undef in
#undef out
#undef KERNEL_SIZE
#undef IWIDTH
#undef IHEIGHT
#undef IDEPTH
#undef OWIDTH
#undef OHEIGHT
#undef ODEPTH
#define KERNEL_SIZE 5
#define KERNEL_LEN 25
#define IWIDTH 14
#define IHEIGHT 14
#define IDEPTH 6
#define OWIDTH 10
#define OHEIGHT 10
#define ODEPTH 16
#define WORK_GROUP_DIM_0 16
#define WORK_GROUP_DIM_1 1
#define WORK_GROUP_DIM_2 1
#define KERNEL_NAME conv3
#define in buf2
#define out buf3
#define KERNEL_PARAM 
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
#undef in
#undef out
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
#define KERNEL_SIZE 2
#define IWIDTH 10
#define IHEIGHT 10
#define IDEPTH 16
#define OWIDTH 5
#define OHEIGHT 5
#define ODEPTH 16
#define in buf3
#define out buf4
#ifdef __xilinx__
__attribute__ ((reqd_work_group_size(16, 1, 1)))
#endif
__kernel void max4(
    __global float *weight,
    __global float *offset) {
    int c = get_global_id(0);
    int r = get_global_id(1);

    if (c < OWIDTH && r < ODEPTH * OHEIGHT) {

        // Get the index of the element in output feature map.
        int o = r / OHEIGHT;
        r = r % OHEIGHT;

        float sum = 0.0f;

        for (int x = 0; x < KERNEL_SIZE; ++x) {
            for (int y = 0; y < KERNEL_SIZE; ++y) {
                sum += in[(o * IHEIGHT + r * KERNEL_SIZE + x) * IWIDTH + c * KERNEL_SIZE + y];
            }
        }

        sum = sum * weight[o] + offset[o];

        // Get the output index.
        int outIdx = (o * OHEIGHT + r) * OWIDTH + c;
        out[outIdx] = sigmod(sum);
    }
}

#undef in
#undef out
#undef KERNEL_SIZE
#undef IWIDTH
#undef IHEIGHT
#undef IDEPTH
#undef OWIDTH
#undef OHEIGHT
#undef ODEPTH
#define KERNEL_SIZE 5
#define KERNEL_LEN 25
#define IWIDTH 5
#define IHEIGHT 5
#define IDEPTH 16
#define OWIDTH 1
#define OHEIGHT 1
#define ODEPTH 120
#define WORK_GROUP_DIM_0 16
#define WORK_GROUP_DIM_1 1
#define WORK_GROUP_DIM_2 1
#define KERNEL_NAME conv5
#define in buf4
#define out buf5
#define KERNEL_PARAM 
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
#undef in
#undef out
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
#define IN_SIZE 120
#define OUT_SIZE 84
#define BUF_SIZE 10
#define in buf5
#define out buf6
#ifdef __xilinx__
__attribute__((reqd_work_group_size(16, 1, 1)))
#endif
__kernel void full6(
    
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
#undef in
#undef out
#undef IN_SIZE
#undef OUT_SIZE
#undef BUF_SIZE
#define IN_SIZE 84
#define OUT_SIZE 10
#define BUF_SIZE 14
#define in buf6
#ifdef __xilinx__
__attribute__((reqd_work_group_size(10, 1, 1)))
#endif
__kernel void rbf(
        __global float *out,
    __global float *weight,
    __global float *offset
    ) {
    int o = get_global_id(0);
    float inBuf[BUF_SIZE];
    float weightBuf[BUF_SIZE];
    if (o < OUT_SIZE) {
        float sum = 0.0f; 
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
                float diff = weightBuf[j] - inBuf[j];
                sum += diff * diff;
            }
        }
        out[o] = sum;
    }
}
#undef in
#undef IN_SIZE
#undef OUT_SIZE
#undef BUF_SIZE
