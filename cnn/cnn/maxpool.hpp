#ifndef MAXPOOL_HEADER
#define MAXPOOL_HEADER

#include "layer.hpp"

namespace cnn {
    class MaxPoolLayer : public Layer {
    public:

        MaxPoolLayer(const LayerParam &params,
            const vec &weight,
            const vec &offset,
            const cl_context &context,
            const cl_program &program,
            const cl_mem &clIn
            ) : Layer(params, weight, offset, context, program, clIn),
            poolSize(params.kernelSize) {

            assert(params.iDepth == params.oDepth);
            assert((params.iWidth / params.kernelSize) == params.oWidth);
            assert((params.iHeight / params.kernelSize) == params.oHeight);
            assert((params.iWidth % params.kernelSize) == 0);
            assert((params.iHeight % params.kernelSize) == 0);
            assert(offset.size() == params.oDepth);
            assert(offset.size() == params.oDepth);

            //Initialize OpenCL.
            initOpenCL(context, program, clIn, params.kernelName);
        }

        virtual ~MaxPoolLayer() {
        }

        virtual unsigned long long forwardCPU(const vec &in) {

            clock_t start = clock(), diff;

            // For each output feature map.
            for (size_t o = 0; o < oDepth; ++o) {
                // For each element in the output feature map.
                for (size_t r = 0; r < oHeight; ++r) {
                    for (size_t c = 0; c < oWidth; ++c) {
                        float sum = 0.0f;
                        for (size_t x = 0; x < poolSize; ++x) {
                            for (size_t y = 0; y < poolSize; ++y) {
                                sum += in[(o * iHeight + r * poolSize + x) * iWidth + c * poolSize + y];
                            }
                        }
                        out[(o * oHeight + r) * oWidth + c] = sigmod(sum * weight[o] + offset[o]);
                    }
                }
            }

            diff = clock() - start;
            int msec = diff * 1000 / CLOCKS_PER_SEC;

            return (unsigned long long)msec;

        }

        // Forward with OpenCL.
        virtual unsigned long long forwardCL(cl_command_queue &queue) {
            cl_int err;
            cl_event event;
            cl_ulong t1, t2;

            // Prepare the NDRange.
            size_t global[] = {
                closestMultiple(workGroupSize[0], oWidth),
                closestMultiple(workGroupSize[1], oDepth * oHeight),
                workGroupSize[2]
            };

            // Enqueue the kernel.
            err = clEnqueueNDRangeKernel(queue,
                kernel,
                3,
                NULL,
                global,
                workGroupSize,
                0,
                NULL,
                &event);
            handleError(err, "Failed enqueuing kernel. ");

            clWaitForEvents(1, &event);

            err = clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &t1, NULL);
            err = clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &t2, NULL);
            handleError(err, "Failed timing the kernel. ");

            return t2 - t1;
        }

    private:

        // Pool size.
        size_t poolSize;
    };
}

#endif