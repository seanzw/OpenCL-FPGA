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

        // Forward with OpenCL.
        virtual unsigned long long forwardCL(cl_command_queue &queue) {

            cl_int err;
            cl_event event;
            cl_ulong t1, t2;

            // Prepare the NDRange.
            size_t global[] = {
                closestMultiple(workGroupSize[0], oWidth * oDepth * oHeight),
                workGroupSize[1],
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
    };
}

#endif