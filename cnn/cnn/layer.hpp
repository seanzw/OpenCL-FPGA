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
            const cl_context &context
            ) : iWidth(params.iWidth),
            iHeight(params.iHeight),
            iDepth(params.iDepth),
            oWidth(params.oWidth),
            oHeight(params.oHeight),
            oDepth(params.oDepth),
            flag(params.flag) {

            out.resize(oWidth * oHeight * oDepth);

            workGroupSize[0] = params.workGroupSize[0];
            workGroupSize[1] = params.workGroupSize[1];
            workGroupSize[2] = params.workGroupSize[2];

            initOpenCL(context);
        }

        virtual ~Layer() {
            if (flag & BACK) {
                clReleaseMemObject(clOut);
            }
            clReleaseKernel(kernel);
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

        size_t workGroupSize[3];

        // Whether this is the first layer or the last layer.
        const Flag flag;

        // Sigmod function.
        float sigmod(float i) {
            return 1.0f / (1.0f + expf(-i));
        }

        void initOpenCL(const cl_context &context) {
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
        }

    };
}

#endif