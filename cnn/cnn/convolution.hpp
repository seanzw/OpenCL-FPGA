#ifndef CONVOLUTION_HEADER
#define CONVOLUTION_HEADER

#include "layer.hpp"

#include <string>
#include <cstring>

namespace cnn {
    class ConvolutionLayer : public Layer {
    public:

        ConvolutionLayer(size_t iWidth, size_t iHeight, size_t iDepth,
            size_t kernelSize, size_t oDepth,
            const vec &weight, const vec &offset,
            DeviceType type, const std::string &fn,
            const std::string kernelName
            ) : Layer(iWidth, iHeight, iDepth, iWidth - kernelSize + 1, iHeight - kernelSize + 1, oDepth, weight, offset, type),
            kernelSize(kernelSize), kernelName(kernelName) {

            // Resize the output buffer.
            output.resize(oDepth * oWidth * oHeight);

            // Resize the input buffer.
            inputBuffer.resize(kernelSize * kernelSize);

            // Initialize OpenCL.
            if (type != CPU) {
                initOpenCL(fn);
            }
        }

        virtual ~ConvolutionLayer() {
            clReleaseMemObject(clIn);
            clReleaseMemObject(clWeight);
            clReleaseMemObject(clOffset);
            clReleaseMemObject(clOut);
            clReleaseKernel(kernel);
            clReleaseProgram(program);
            clReleaseCommandQueue(queue);
            clReleaseContext(context);
        }

        // Forward.
        virtual unsigned long long forward(const vec &in) {
            switch (type) {
            case cnn::CPU:
                return forwardCPU(in);
                break;
            case cnn::GPU:
            case cnn::FPGA:
                return forwardCL(in);
                break;
            default:
                return forwardCPU(in);
                break;
            }
        }

        // Forward with CPU.
        unsigned long long forwardCPU(const vec &in) {

            clock_t start = clock(), diff;
            
            // Clear the output buffer.
            std::fill(output.begin(), output.end(), 0.0f);

            // For each output feature map.
            for (size_t o = 0; o < oDepth; ++o) {
                // For each input feature map.
                for (size_t i = 0; i < iDepth; ++i) {
                    // For each element in the output feature map.
                    for (size_t r = 0; r < oHeight; ++r) {
                        for (size_t c = 0; c < oWidth; ++c) {
                            getInput(i, r, c, in);
                            output[getOutputIdx(o, r, c)] += convolution(getWeightBase(i, o));
                        }
                    }
                }

                // Activate function.
                for (size_t r = 0; r < oHeight; ++r) {
                    for (size_t c = 0; c < oWidth; ++c) {
                        size_t idx = getOutputIdx(o, r, c);
                        output[idx] = sigmod(output[idx] + offset[o]);
                    }
                }
            }

            diff = clock() - start;
            int msec = diff * 1000 / CLOCKS_PER_SEC;

            return (unsigned long long)msec;
        }

        // Forward with OpenCL on GPU.
        unsigned long long forwardCL(const vec &in) {

            cl_int err;

            // Copy the input.
            err = clEnqueueWriteBuffer(queue,
                clIn,
                CL_TRUE,
                0,
                iWidth * iHeight * iDepth * sizeof(cl_float),
                &in[0],
                0,
                NULL,
                NULL);
            handleError(err, "Failed writing clIn.");

            // Prepare the NDRange.
            size_t local[] = {
                16,
                1,
                1
            };
            size_t global[] = {
                closestMultiple(local[0], oWidth),
                closestMultiple(local[1], oDepth * oHeight),
                1
            };

            cl_ulong t = runAndTimeKernel(queue, kernel, 3, global, local);
            err = clEnqueueReadBuffer(queue,
                clOut,
                CL_TRUE,
                0,
                oWidth * oHeight * oDepth * sizeof(cl_float),
                (void *)&output[0],
                0,
                NULL,
                NULL);
            if (err != CL_SUCCESS) {
                std::cerr << "Failed reading the output." << std::endl;
                std::cerr << readable_status(err);
                exit(-1);
            }

            return t;
        }

        // Prepare the input buffer.
        inline void getInput(size_t i, size_t r, size_t c, const vec &in) {
            size_t idx = 0;
            for (size_t x = 0; x < kernelSize; ++x) {
                for (size_t y = 0; y < kernelSize; ++y) {
                    inputBuffer[idx++] = in[i * iWidth * iHeight + (r + x) * iWidth + c + y];
                }
            }
        }

        // Get the output feature map element index.
        inline size_t getOutputIdx(size_t o, size_t r, size_t c) {
            return o * oWidth * oHeight + r * oWidth + c;
        }

        // Get the base index of the weight.
        inline size_t getWeightBase(size_t i, size_t o) {
            return (o * iDepth + i) * kernelSize * kernelSize;
        }

        // Do the convolution with weight and the input buffer.
        float convolution(size_t weightBase) {
            float sum = 0.0f;
            for (size_t i = 0; i < kernelSize * kernelSize; ++i) {
                sum += weight[weightBase + i] * inputBuffer[i];
            }
            return sum;
        }

        size_t kernelSize;

        // Buffer for convolution.
        vec inputBuffer;

        // For OpenCL.
        cl_context context;
        cl_command_queue queue;
        cl_program program;
        cl_kernel kernel;

        cl_mem clIn;
        cl_mem clWeight;
        cl_mem clOffset;
        cl_mem clOut;

        const std::string kernelName;

        // Initialize the OpenCL.
        void initOpenCL(const std::string &fn) {
            cl_platform_id platforms[2];
            cl_device_id devices[2];
            cl_int err;

            // Choose the first platform.
            err = clGetPlatformIDs(1, platforms, NULL);

            // Switch between FPGA device and GPU device.
            err = clGetDeviceIDs(platforms[0],
                type == FPGA ? CL_DEVICE_TYPE_ACCELERATOR : CL_DEVICE_TYPE_GPU,
                1,
                devices,
                NULL);

            printDeviceInfo(std::cout, devices[0]);

            cl_context_properties properties[] = {
                CL_CONTEXT_PLATFORM, (cl_context_properties)platforms[0], 0
            };

            context = clCreateContext(
                properties,
                1,
                devices,
                NULL,
                NULL,
                &err);
            handleError(err, "Failed creating context. ");
            clRetainContext(context);

            queue = clCreateCommandQueue(
                context,
                devices[0],
                CL_QUEUE_PROFILING_ENABLE,
                &err);
            handleError(err, "Failed creating command queue. ");
            clRetainCommandQueue(queue);

            if (type == FPGA) {
                program = buildProgramFromBinary(fn.c_str(), context, devices[0]);
            }
            else {
                program = buildProgramFromSource(fn.c_str(), context, devices[0]);
            }
            err = clRetainProgram(program);
            handleError(err, "Failed retaining program. ");

            // Allocate memory on device.
            clIn = clCreateBuffer(
                context,
                CL_MEM_READ_ONLY,
                iWidth * iHeight * iDepth * sizeof(cl_float),
                NULL,
                &err);
            handleError(err, "Failed creating clIn");
            err = clRetainMemObject(clIn);
            handleError(err, "Failed retaining clIn");

            clWeight = clCreateBuffer(
                context,
                CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                kernelSize * kernelSize * iDepth * oDepth * sizeof(cl_float),
                const_cast<void *>(static_cast<const void *>(&weight[0])),
                &err);
            handleError(err, "Failed creating clWeight");
            err = clRetainMemObject(clWeight);
            handleError(err, "Failed retaining clWeight");

            clOffset = clCreateBuffer(
                context,
                CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                oDepth * sizeof(cl_float),
                const_cast<void *>(static_cast<const void *>(&offset[0])),
                &err);
            handleError(err, "Failed creating clOffset");
            err = clRetainMemObject(clOffset);
            handleError(err, "Failed retaining clOffset");

            clOut = clCreateBuffer(
                context,
                CL_MEM_WRITE_ONLY,
                oDepth * oHeight * oWidth * sizeof(cl_float),
                NULL,
                &err);
            handleError(err, "Failed creating clOut");
            err = clRetainMemObject(clOut);
            handleError(err, "Failed retaining clOut");

            // Set the arguments for the kernel.
            kernel = clCreateKernel(program, kernelName.c_str(), &err);
            handleError(err, "Failed creating kernel. ");
            err = clRetainKernel(kernel);
            handleError(err, "Failed retaining kernel. ");

            err = clSetKernelArg(kernel, 0, sizeof(cl_mem), &clIn);
            handleError(err, "Failed setting kernel arg: clIn. ");

            err = clSetKernelArg(kernel, 1, sizeof(cl_mem), &clWeight);
            handleError(err, "Failed setting kernel arg: clWeight. ");

            err = clSetKernelArg(kernel, 2, sizeof(cl_mem), &clOffset);
            handleError(err, "Failed setting kernel arg: clOffset. ");

            err = clSetKernelArg(kernel, 3, sizeof(cl_mem), &clOut);
            handleError(err, "Failed setting kernel arg: clOut. ");
        }
    };

    // Helper function to create a convolution layer from xml file.
    ConvolutionLayer createConvolutionLayerFromXML(rapidxml::xml_node<> *root, DeviceType type, const std::string &clFile) {

        // Get the parameters for the convolutional layer.
        int iWidth = getInt(root, "iWidth");
        int iHeight = getInt(root, "iHeight");
        int iDepth = getInt(root, "iDepth");
        int kernelSize = getInt(root, "kernelSize");
        int oDepth = getInt(root, "oDepth");

        // Get the kernel name.
        std::string kernelName = getString(root, "kernelName");

        // Create the weight vector.
        cnn::vec weight;
        getAllItem(root->first_node("weight"), weight);
        assert(weight.size() == oDepth * iDepth * kernelSize * kernelSize);

        // Create the offset vector.
        cnn::vec offset;
        for (rapidxml::xml_node<> *node = root->first_node("offset")->first_node(); node; node = node->next_sibling()) {
            offset.push_back((float)std::atof(node->value()));
        }
        assert(offset.size() == oDepth);

        cnn::ConvolutionLayer layer(iWidth,
            iHeight,
            iDepth,
            kernelSize,
            oDepth,
            weight,
            offset,
            type,
            type == FPGA ? clFile : kernelName + ".cl",
            kernelName
            );
        return layer;
    }
}

#endif