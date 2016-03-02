float sigmod(float in) {
    return 1.0f / (1.0f + exp(-in)); 
}
__global float buf1[4704];
#define KERNEL_SIZE 5
#define KERNEL_LEN 25
#define IWIDTH 32
#define IHEIGHT 32
#define IDEPTH 1
#define OWIDTH 28
#define OHEIGHT 28
#define ODEPTH 6
#define out buf1
#ifdef __xilinx__
__attribute__ ((reqd_work_group_size(16, 1, 1)))
#endif
__kernel void conv1(
    __global float *in,

    __constant float *weight,
    __constant float *offset
    ) {
    
    #ifdef __xilinx__
    __attribute__((xcl_pipeline_workitems))
    #endif
    int c = get_global_id(0);
    int r = get_global_id(1);
    
    if (c < OWIDTH && r < ODEPTH * OHEIGHT) {

        // Get the index of the element in output feature map.
        int o = r / OHEIGHT;
        r = r % OHEIGHT; 

        float sum = 0.0f;

        // For each input feature map.
        #ifdef __xilinx__
        __attribute__((xcl_pipeline_loop))
        #endif
        for (int i = 0; i < IDEPTH; ++i) {
            
            float inputBuf[KERNEL_LEN];
            float weightBuf[KERNEL_LEN];
            int idx = 0;
            int weightBase = (o * IDEPTH + i) * KERNEL_LEN;
            #ifdef __xilinx__
            __attribute__((opencl_unroll_hint))
            #endif
            for (int x = 0; x < KERNEL_SIZE; ++x) {
                #ifdef __xilinx__
                __attribute__((opencl_unroll_hint))
                #endif
                for (int y = 0; y < KERNEL_SIZE; ++y) {
                    inputBuf[idx] = in[(i * IHEIGHT + r + x) * IWIDTH + c + y];
                    weightBuf[idx] = weight[weightBase + idx];
                    idx++;
                }
            }

            #ifdef __xilinx__
            __attribute__((opencl_unroll_hint))
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
#undef out
#undef KERNEL_SIZE
#undef KERNEL_LEN
#undef IWIDTH
#undef IHEIGHT
#undef IDEPTH
#undef OWIDTH
#undef OHEIGHT
#undef ODEPTH
#define KERNEL_SIZE 2
#define IWIDTH 28
#define IHEIGHT 28
#define IDEPTH 6
#define OWIDTH 14
#define OHEIGHT 14
#define ODEPTH 6
#define in buf1
#ifdef __xilinx__
__attribute__ ((reqd_work_group_size(16, 1, 1)))
#endif
__kernel void max1(    __global float *out) {
    int c = get_global_id(0);
    int r = get_global_id(1);
    
    if (c < OWIDTH && r < ODEPTH * OHEIGHT) {
    
        // Get the index of the element in output feature map.
        int o = r / OHEIGHT;
        r = r % OHEIGHT;
    
        float max = 0.0f;
    
        for (int x = 0; x < KERNEL_SIZE; ++x) {
            for (int y = 0; y < KERNEL_SIZE; ++y) {
                float tmp = in[(o * IHEIGHT + r * KERNEL_SIZE + x) * IWIDTH + c * KERNEL_SIZE + y];
                if (tmp > max) {
                    max = tmp;
                }
            }
        }
    
        // Get the output index.
        int outIdx = (o * OHEIGHT + r) * OWIDTH + c;
        out[outIdx] = max;
    }
}

#undef in
#undef KERNEL_SIZE
#undef IWIDTH
#undef IHEIGHT
#undef IDEPTH
#undef OWIDTH
#undef OHEIGHT
#undef ODEPTH
