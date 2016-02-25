#define KERNEL_SIZE 5
#define KERNEL_LEN 25
#define IWIDTH 32
#define IHEIGHT 32
#define IDEPTH 1
#define OWIDTH 28
#define OHEIGHT 28
#define ODEPTH 6

float sigmod(float in) {
    return 1.0f / (1.0f + exp(-in));
}

#ifdef __xilinx__
__attribute__ ((reqd_work_group_size(16, 16, 1)))
#endif
__kernel void convolution_kernel_opt_1(
    __global float *in,
    __global float *weight,
    __global float *offset,
    __global float *out
    ) {

    #ifdef __xilinx__
    __attribute__((xcl_pipeline_workitems))
    #endif
    
    int c = get_global_id(0);
    int r = get_global_id(1);
    
    if (c < OWIDTH && r < ODEPTH * OHEIGHT) {

        // Get the index of the element in output feature map.
        int o = r / OHEIGHT;
        int r = r % OHEIGHT;

        float sum = 0.0f;

        float inputBuf[KERNEL_LEN];
        float weightBuf[KERNEL_LEN];

        // For each input feature map.            
        #ifdef __xilinx__
        __attribute__((xcl_pipeline_loop))
        #endif
        for (int i = 0; i < IDEPTH; ++i) {
            
            // Prepare the input buffer and weight buffer.
            // Copy them into the private memory.

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

            // Compute the convolution.
            #ifdef __xilinx__
            __attribute__((opencl_unroll_hint))
            #endif
            for (int x = 0; x < KERNEL_LEN; ++x) {
                sum += inputBuf[x] * weightBuf[x];
            }
        }

        // Get the output index.
        int outIdx = (o * OHEIGHT + r) * OWIDTH + c;
        out[outIdx] = sigmod(sum + offset[outIdx]);
    } 
}