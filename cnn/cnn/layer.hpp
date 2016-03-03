#ifndef LAYER_HEADER
#define LAYER_HEADER

#include "util.hpp"
#include <cmath>

namespace cnn {

    enum LayerType {
        CONV,
        SUB,
        FULL
    };

#define INNER (0)
#define FRONT (1)
#define BACK  (1 << 1)
    typedef size_t Flag;

    struct LayerParam {
        LayerType type;
        std::string kernelName;
        size_t workGroupSize[3];
        size_t iWidth;
        size_t iHeight;
        size_t iDepth;
        size_t kernelSize;
        size_t oWidth;
        size_t oHeight;
        size_t oDepth;
        Flag flag;
    };

    class Layer {
    public:

        Layer(const LayerParam &params,
            const vec &weight,
            const vec &offset,
            const cl_context &context,
            const cl_program &program,
            const cl_mem &clIn
            ) : iWidth(params.iWidth),
            iHeight(params.iHeight),
            iDepth(params.iDepth),
            oWidth(params.oWidth),
            oHeight(params.oHeight),
            oDepth(params.oDepth),
            flag(params.flag),
            weight(weight),
            offset(offset) {

            out.resize(oWidth * oHeight * oDepth);

            workGroupSize[0] = params.workGroupSize[0];
            workGroupSize[1] = params.workGroupSize[1];
            workGroupSize[2] = params.workGroupSize[2];

            initOpenCL(context, program, clIn, params.kernelName);
        }

        virtual ~Layer() {
            if (flag & BACK) {
                clReleaseMemObject(clOut);
            }
            clReleaseKernel(kernel);
            clReleaseMemObject(clWeight);
            clReleaseMemObject(clOffset);
        }
        
        virtual unsigned long long forwardCPU(const vec &in) = 0;
        virtual unsigned long long forwardCL(cl_command_queue &queue) = 0;


        friend class CNN;

    protected:

        size_t iWidth;
        size_t iHeight;
        size_t iDepth;

        size_t oWidth;
        size_t oHeight;
        size_t oDepth;

        // For CPU.
        vec out;

        // For OpenCL.
        cl_kernel kernel;
        cl_mem clOut;

        // Buffer.
        vec weight;
        vec offset;

        // For OpenCL.
        cl_mem clWeight;
        cl_mem clOffset;

        size_t workGroupSize[3];

        // Whether this is the first layer or the last layer.
        const Flag flag;

        // Sigmod function.
        float sigmod(float i) {
            return 1.0f / (1.0f + expf(-i));
        }

        void initOpenCL(const cl_context &context,
            const cl_program &program,
            const cl_mem &clIn,
            const std::string &kernelName
            ) {
            cl_int err;
            if (flag & BACK) {
                clOut = clCreateBuffer(
                    context,
                    CL_MEM_WRITE_ONLY,
                    oDepth * oHeight * oWidth * sizeof(cl_float),
                    NULL,
                    &err);
                handleError(err, "Failed creating clOut");
                err = clRetainMemObject(clOut);
                handleError(err, "Failed retaining clOut");
            }

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