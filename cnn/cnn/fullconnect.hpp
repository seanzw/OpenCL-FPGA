#ifndef FULLCONNECT_HEADER
#define FULLCONNECT_HEADER

#include "layer.hpp"

namespace cnn {
    class FullConnectLayer : public Layer {
    public:

        FullConnectLayer(const LayerParam &params,
            const vec &weight,
            const vec &offset,
            const cl_context &context,
            const cl_program &program,
            const cl_mem &clIn
            ) : Layer(params, weight, offset, context, program, clIn) {

            assert(weight.size() == (iWidth * iHeight * iDepth * oWidth * oHeight * oDepth));
            assert(offset.size() == (oWidth * oHeight * oDepth));

            // Initialize OpenCL.
            initOpenCL(context, program, clIn, params.kernelName);

            // Prepare the ND-Range.
            global[0] = closestMultiple(workGroupSize[0], oWidth * oDepth * oHeight);
            global[1] = workGroupSize[1];
            global[2] = workGroupSize[2];
        }

        virtual ~FullConnectLayer() {
            clReleaseMemObject(clWeight);
            clReleaseMemObject(clOffset);
        }

        virtual unsigned long long forwardCPU(const vec &in) {

            clock_t start = clock(), diff;

            // Clear the output buffer.
            std::fill(out.begin(), out.end(), 0.0f);

            // For each output element.
            for (size_t o = 0; o < out.size(); ++o) {
                size_t weightBase = o * iWidth * iHeight * iDepth;
                // For each input element.
                float sum = 0.0f;
                for (size_t i = 0; i < in.size(); ++i) {
                    sum += weight[weightBase + i] * in[i];
                }
                out[o] = sigmod(sum + offset[o]);
            }

            diff = clock() - start;
            int msec = diff * 1000 / CLOCKS_PER_SEC;

            return (unsigned long long)msec;
        }


    };
}

#endif