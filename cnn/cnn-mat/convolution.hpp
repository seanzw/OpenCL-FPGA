#ifndef CONVOLUTION_H
#define CONVOLUTION_H

#include "util.hpp"

namespace cnn {
    class ConvolutionLayer {
    public:
        ConvolutionLayer(
            size_t iWidth,
            size_t iHeight,
            size_t iDepth,
            size_t kernelSize,
            size_t oDepth
            ) {
            this->iWidth = iWidth;
            this->iHeight = iHeight;
            this->iDepth = iDepth;
            this->oWidth = iWidth - kernelSize + 1;
            this->oHeight = iHeight - kernelSize + 1;
            this->oDepth = oDepth;
            this->kernelSize = kernelSize;
            this->M = oDepth;
            this->N = kernelSize * kernelSize * iDepth;
            this->P = oWidth * oHeight;

            global[0] = 1;
            global[1] = 1;
            global[2] = 1;

            local[0] = 1;
            local[1] = 1;
            local[2] = 1;

            _A.resize(M * N);
            _B.resize(N * P);
            _C.resize(M * P);

            for (int i = 0; i < _A.size(); ++i) {
                _A[i] = rand();
            }

            for (int i = 0; i < _B.size(); ++i) {
                _B[i] = rand();
            }

            cl_int err;
            // Choose the first platform.
            err = clGetPlatformIDs(1, &platform, NULL);

            // Choose the first device.
            err = clGetDeviceIDs(platform,
                CL_DEVICE_TYPE_ALL,
                1,
                &device,
                NULL);

            printDeviceInfo(std::cout, device);

            cl_context_properties properties[] = {
                CL_CONTEXT_PLATFORM, (cl_context_properties)platform, 0
            };

            context = clCreateContext(
                properties,
                1,
                &device,
                NULL,
                NULL,
                &err);
            handleError(err, "Failed creating context. ");
            clRetainContext(context);

            queue = clCreateCommandQueue(
                context,
                device,
                CL_QUEUE_PROFILING_ENABLE,
                &err);
            handleError(err, "Failed creating command queue. ");
            clRetainCommandQueue(queue);

            A = clCreateBuffer(
                context,
                CL_MEM_READ_ONLY,
                M * N * sizeof(cl_float),
                NULL,
                &err);
            handleError(err, "Failed creating A");
            err = clRetainMemObject(A);
            handleError(err, "Failed retaining A");

            B = clCreateBuffer(
                context,
                CL_MEM_READ_ONLY,
                N * P * sizeof(cl_float),
                NULL,
                &err);
            handleError(err, "Failed creating B");
            err = clRetainMemObject(B);
            handleError(err, "Failed retaining B");

            C = clCreateBuffer(
                context,
                CL_MEM_WRITE_ONLY,
                M * P * sizeof(cl_float),
                NULL,
                &err);
            handleError(err, "Failed creating C");
            err = clRetainMemObject(C);
            handleError(err, "Failed retaining C");

            //program = buildProgramFromSource("conv1.cl", context, device);
            program = buildProgramFromBinary("alpha.xclbin", context, device);

            err = clRetainProgram(program);
            handleError(err, "Failed retaining program.");

            kernel = clCreateKernel(program, "conv1", &err);
            handleError(err, "Failed creating kernel.");

            clSetKernelArg(kernel, 0, sizeof(cl_mem), &A);
            clSetKernelArg(kernel, 1, sizeof(cl_mem), &B);
            clSetKernelArg(kernel, 2, sizeof(cl_mem), &C);
        }

        unsigned long long forwardCL() {
            
            // Prepare A and B.
            cl_uint err;
            cl_event event;
            cl_ulong t1, t2;
            err = clEnqueueWriteBuffer(queue,
                A,
                CL_TRUE,
                0,
                M * N * sizeof(cl_float),
                (void*)&_A[0],
                0,
                NULL,
                NULL);
            handleError(err, "Failed creating A");

            err = clEnqueueWriteBuffer(queue,
                B,
                CL_TRUE,
                0,
                P * N * sizeof(cl_float),
                (void*)&_B[0],
                0,
                NULL,
                NULL);
            handleError(err, "Failed creating B");

            err = clEnqueueNDRangeKernel(queue,
                kernel,
                3,
                NULL,
                global,
                local,
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
        size_t iWidth, iHeight, iDepth;
        size_t oWidth, oHeight, oDepth;
        size_t kernelSize;
        size_t M, N, P;

        vec _A, _B, _C;

        cl_mem A;
        cl_mem B;
        cl_mem C;

        size_t global[3];
        size_t local[3];

        cl_platform_id platform;
        cl_device_id device;
        cl_context context;
        cl_command_queue queue;
        cl_program program;
        cl_kernel kernel;
    };
}

#endif