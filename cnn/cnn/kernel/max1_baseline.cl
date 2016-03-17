float sigmod(float in) {
    return 1.0f / (1.0f + exp(-in)); 
}
#define KERNEL_SIZE 2
#define IWIDTH 28
#define IHEIGHT 28
#define IDEPTH 6
#define OWIDTH 14
#define OHEIGHT 14
#define ODEPTH 6
#ifdef __xilinx__
__attribute__ ((reqd_work_group_size(16, 1, 1)))
#endif
__kernel void max1(__global float *in,
    __global float *out,
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

#undef KERNEL_SIZE
#undef IWIDTH
#undef IHEIGHT
#undef IDEPTH
#undef OWIDTH
#undef OHEIGHT
#undef ODEPTH
