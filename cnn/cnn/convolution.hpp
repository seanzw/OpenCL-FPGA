#ifndef CONVOLUTION_HEADER
#define CONVOLUTION_HEADER

#include "layer.hpp"

#include <string>

namespace cnn {
    class ConvolutionLayer : public Layer {
    public:

        ConvolutionLayer(size_t iWidth, size_t iHeight, size_t iDepth,
            size_t kernelSize, size_t oDepth,
            const vec &weight, const vec &offset
            ) : Layer(iWidth, iHeight, iDepth, iWidth - kernelSize + 1, iHeight - kernelSize + 1, oDepth, weight, offset),
            kernelSize(kernelSize) {

            // Resize the output buffer.
            output.resize(oDepth * oWidth * oHeight);

            // Resize the input buffer.
            inputBuffer.resize(kernelSize * kernelSize);

            // Initialize OpenCL.
            initOpenCL();
        }

        // Forward.
        virtual void forward(const vec &in) {
            forwardCPU(in);
        }

        // Forward with CPU.
        void forwardCPU(const vec &in) {

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
        }

        // Forward with OpenCL on GPU.
        void forwardGPU(const vec &in) {

            // Allocate memory on device.
            cl::Buffer clIn(context, CL_MEM_READ_ONLY, iWidth * iHeight * iDepth * sizeof(cl_float));
            cl::Buffer clWeight(context, CL_MEM_READ_ONLY, kernelSize * kernelSize * iDepth * oDepth * sizeof(cl_float));
            cl::Buffer clOffset(context, CL_MEM_READ_ONLY, oDepth * sizeof(cl_float));
            cl::Buffer clOut(context, CL_MEM_WRITE_ONLY, oWidth * oHeight * oDepth * sizeof(cl_float));

            // Set the arguments for the kernel.
            std::string kernelName = "forwardGPU";
            cl::Kernel kernel(program, kernelName.c_str());
            kernel.setArg<cl::Memory>(0, clIn);
            kernel.setArg<cl::Memory>(1, clWeight);
            kernel.setArg<cl::Memory>(2, clOffset);
            kernel.setArg<cl::Memory>(3, clOut);
            kernel.setArg<int>(4, (int)iWidth);
            kernel.setArg<int>(5, (int)iHeight);
            kernel.setArg<int>(6, (int)iDepth);
            kernel.setArg<int>(7, (int)oWidth);
            kernel.setArg<int>(8, (int)oHeight);
            kernel.setArg<int>(9, (int)oDepth);
            kernel.setArg<int>(10, (int)kernelSize);

            // Copy the data from host to device.
            queue.enqueueWriteBuffer(clIn, CL_TRUE, 0, iWidth * iHeight * iDepth * sizeof(cl_float), &in[0]);
            queue.enqueueWriteBuffer(clWeight, CL_TRUE, 0, kernelSize * kernelSize * iDepth * oDepth * sizeof(cl_float), &weight[0]);
            queue.enqueueWriteBuffer(clOffset, CL_TRUE, 0, oDepth * sizeof(cl_float), &offset[0]);

            // Prepare the NDRange.
            int items = 16;
            cl::NDRange global(closestMultiple(items, (int)oWidth),
                closestMultiple(items, (int)(oDepth * oHeight)));
            cl::NDRange local(items, items);
            cl_ulong t = runAndTimeKernel(kernel, queue, global, local);

            queue.enqueueReadBuffer(clOut, CL_TRUE, 0, oWidth * oHeight * oDepth * sizeof(cl_float), &output[0]);
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
        cl::Context context;
        cl::CommandQueue queue;
        cl::Program program;

        // Initialize the OpenCL.
        void initOpenCL() {
            std::vector<cl::Platform> platforms;
            std::vector<cl::Device> devices;

            cl::Platform::get(&platforms);
            platforms[0].getDevices(CL_DEVICE_TYPE_GPU, &devices);

            // Get the first device.
            context = cl::Context(devices);
            queue = cl::CommandQueue(context, devices[0], CL_QUEUE_PROFILING_ENABLE);
            program = buildProgram("convolution.cl", context, devices);

        }
    };

    // Helper function to create a convolution layer from xml file.
    ConvolutionLayer createConvolutionLayerFromXML(const std::string &fn) {
        std::string str = fileToString(fn);
        char *text = new char[str.size() + 1];
        stdext::checked_array_iterator<char *> checked(text, str.size() + 1);
        std::copy(str.begin(), str.end(), checked);
        text[str.size()] = '\0';

        // Parse the xml file.
        rapidxml::xml_document<> doc;
        doc.parse<0>(text);

        rapidxml::xml_node<> *root = doc.first_node("ConvolutionalLayer");

        auto getInt = [](rapidxml::xml_node<> *root, const char *name) -> int {
            rapidxml::xml_node<> *node = root->first_node(name);
            return std::stoi(node->value());
        };

        // Get the parameters for the convolutional layer.
        int iWidth = getInt(root, "iWidth");
        int iHeight = getInt(root, "iHeight");
        int iDepth = getInt(root, "iDepth");
        int kernelSize = getInt(root, "kernelSize");
        int oDepth = getInt(root, "oDepth");

        // Create the weight vector.
        cnn::vec weight;
        getAllItem(root->first_node("weight"), weight);
        assert(weight.size() == oDepth * iDepth * kernelSize * kernelSize);

        // Create the offset vector.
        cnn::vec offset;
        for (rapidxml::xml_node<> *node = root->first_node("offset")->first_node(); node; node = node->next_sibling()) {
            offset.push_back(std::stof(node->value()));
        }
        assert(offset.size() == oDepth);

        delete[] text;

        cnn::ConvolutionLayer layer(iWidth, iHeight, iDepth, kernelSize, oDepth, weight, offset);
        return layer;
    }
}

#endif