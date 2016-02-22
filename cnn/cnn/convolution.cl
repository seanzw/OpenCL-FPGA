#define MAX_KERNEL_LENGTH 25

float sigmod(float in) {
    return 1.0f / (1.0f + exp(-in));
}

__kernel void forwardGPU(
    __global float *in,
    __global float *weight,
    __global float *offset,
    __global float *out,
    int iWidth,
    int iHeight,
    int iDepth,
    int oWidth,
    int oHeight,
    int oDepth,
    int kernelSize
    ) {
    
    int c = get_global_id(0);
    int r = get_global_id(1);
    
    if (c < oWidth && r < oDepth * oHeight) {

        // Get the index of the element in output feature map.
        int o = r / oHeight;
        int r = r % oHeight;

        float sum = 0.0f;
        int kernelLength = kernelSize * kernelSize;

        // For each input feature map.            
        for (int i = 0; i < iDepth; ++i) {
            
            // Prepare the input buffer and weight buffer.
            // Copy them into the private memory.
            float inputBuf[MAX_KERNEL_LENGTH];
            float weightBuf[MAX_KERNEL_LENGTH];
            int idx = 0;
            int weightBase = (o * iDepth + i) * kernelLength;
            for (int x = 0; x < kernelSize; ++x) {
                for (int y = 0; y < kernelSize; ++y) {
                    inputBuf[idx] = in[(i * iHeight + r + x) * iWidth + c + y];
                    weightBuf[idx] = weight[weightBase + idx];
                    idx++;
                }
            }

            // Compute the convolution.
            for (int x = 0; x < kernelLength; ++x) {
                sum += inputBuf[x] * weightBuf[x];
            }
        }

        // Get the output index.
        int outIdx = (o * oHeight + r) * oWidth + c;
        out[outIdx] = sigmod(sum + offset[outIdx]);
    } 
}