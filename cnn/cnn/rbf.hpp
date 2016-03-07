#include "layer.hpp"

namespace cnn {
    class RBFLayer : public Layer {
    public:
        RBFLayer(const LayerParam &params,
            const vec &weight,
            const vec &offset,
            const cl_context &context,
            const cl_program &program,
            const cl_mem &clIn
            ) : Layer(params, weight, offset, context, program, clIn) {

            assert(weight.size() == (iWidth * iHeight * iDepth * oWidth * oHeight * oDepth));

            // Prepare the ND-Range.
            global[0] = closestMultiple(workGroupSize[0], oWidth * oDepth * oHeight);
            global[1] = workGroupSize[1];
            global[2] = workGroupSize[2];
        }

        virtual ~RBFLayer() {

        }

        // Forward with CPU.
        virtual unsigned long long forwardCPU(const vec &in) {

            clock_t start = clock(), diff;

            // Clear the output buffer.
            std::fill(out.begin(), out.end(), 0.0f);

            for (size_t o = 0; o < out.size(); ++o) {
                for (size_t i = 0; i < in.size(); ++i) {
                    float diff = in[i] - weight[o * in.size() + i];
                    out[o] += diff * diff;
                }
            }

            diff = clock() - start;
            int msec = diff * 1000 / CLOCKS_PER_SEC;

            return (unsigned long long)msec;
        }
    };
}