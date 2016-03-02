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
            ) : Layer(params, context),
            weight(weight),
            offset(offset) {

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

    private:

        // Buffer.
        vec weight;
        vec offset;

        // For OpenCL.
        cl_mem clWeight;
        cl_mem clOffset;

        // Initialize the OpenCL.
        void initOpenCL(const cl_context &context,
            const cl_program &program,
            const cl_mem &clIn,
            const std::string &kernelName
            ) {

            cl_int err;

            clWeight = clCreateBuffer(
                context,
                CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                weight.size() * sizeof(cl_float),
                const_cast<void *>(static_cast<const void *>(&weight[0])),
                &err);
            handleError(err, "Failed creating clWeight. ");
            err = clRetainMemObject(clWeight);
            handleError(err, "Failed retaining clWeight. ");

            clOffset = clCreateBuffer(
                context,
                CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                offset.size() * sizeof(cl_float),
                const_cast<void *>(static_cast<const void *>(&offset[0])),
                &err);
            handleError(err, "Failed creating clOffset");
            err = clRetainMemObject(clOffset);
            handleError(err, "Failed retaining clOffset");

            // Create the kernel and set the arguments.
            kernel = clCreateKernel(program, kernelName.c_str(), &err);
            handleError(err, "Failed creating kernel. ");
            err = clRetainKernel(kernel);
            handleError(err, "Failed retaining kernel. ");

            unsigned int i = 0;
            if (flag & FRONT) {
                err = clSetKernelArg(kernel, i++, sizeof(cl_mem), &clIn);
                handleError(err, "Failed setting kernel arg: clIn. ");
            }

            if (flag & BACK) {
                err = clSetKernelArg(kernel, i++, sizeof(cl_mem), &clOut);
                handleError(err, "Failed setting kernel arg: clOut. ");
            }

            err = clSetKernelArg(kernel, i++, sizeof(cl_mem), &clWeight);
            handleError(err, "Failed setting kernel arg: clWeight. ");

            err = clSetKernelArg(kernel, i++, sizeof(cl_mem), &clOffset);
            handleError(err, "Failed setting kernel arg: clOffset. ");
        }
    };
}

#endif