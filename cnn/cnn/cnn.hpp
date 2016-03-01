#ifndef CNN_HEADER
#define CNN_HEADER

#include "util.hpp"
#include "convolution.hpp"
#include "maxpool.hpp"

#define BUFSIZE (64 * 1024)

namespace cnn {
    class CNN {
    public:

        CNN(const std::string &xmlFileName, const std::string &xclbinFile = "NONE") {

            // Parse the xml file.
            char *buf = new char[BUFSIZE];
            fileToChar(xmlFileName, buf, BUFSIZE);
            
            rapidxml::xml_document<> doc;
            doc.parse<0>(buf);
            rapidxml::xml_node<> *root = doc.first_node();

            // Get the kernel file name.
            std::string kernelFileName(root->first_node("kernelFileName")->value());

            // Get the input size.
            size_t inSize = getSizeT(root, "inSize");

            // Initialize the OpenCL.
            if (xclbinFile == "NONE") {
                initOpenCL(kernelFileName, false, inSize);
            }
            else {
                initOpenCL(xclbinFile, true, inSize);
            }

            // For every layer.
            bool isFront = true;
            for (rapidxml::xml_node<> *layer = root->first_node("layer"); layer; layer = layer->next_sibling("layer")) {
                Flag flag = INNER;
                if (isFront) {
                    flag |= FRONT;
                    isFront = false;
                }
                if (!(layer->next_sibling())) {
                    flag |= BACK;
                }
                layers.push_back(createLayer(layer, flag));
            }

            delete buf;
        }

        ~CNN() {
            for (int i = 0; i < layers.size(); ++i) {
                delete layers[i];
            }
            clReleaseProgram(program);
            clReleaseCommandQueue(queue);
            clReleaseContext(context);
        }

        // Forward with CPU.
        unsigned long long forwardCPU(const vec &in) {

            unsigned long long totalTime = layers[0]->forwardCPU(in);
            for (size_t i = 1; i < layers.size(); ++i) {
                totalTime += layers[i]->forwardCPU(layers[i - 1]->out);
            }

            return totalTime;
        }

        unsigned long long forwardCL(const vec &in) {

            // Prepare the input cl_mem.
            cl_int err;
            err = clEnqueueWriteBuffer(queue,
                clIn,
                CL_TRUE,
                0,
                in.size() * sizeof(cl_float),
                (void *)&in[0],
                0,
                NULL,
                NULL);
            handleError(err, "Failed copy input buffer. ");

            // Enqueue the first kernel.
            unsigned long long totalTime = layers[0]->forwardCL(queue);
            for (size_t i = 1; i < layers.size(); ++i) {
                totalTime += layers[i]->forwardCL(queue);
            }

            // Get the result to the last layer's out vec.
            err = clEnqueueReadBuffer(queue,
                layers[layers.size() - 1]->clOut,
                CL_TRUE,
                0,
                getOutSize() * sizeof(cl_float),
                &(layers[layers.size() - 1]->out[0]),
                0,
                NULL,
                NULL);

            return totalTime;
        }

        // For OpenCL.
        cl_platform_id platform;
        cl_device_id device;
        cl_context context;
        cl_command_queue queue;
        cl_program program;
        cl_mem clIn;

        std::vector<Layer *> layers;

        size_t getInSize() const {
            return layers[0]->iWidth * layers[0]->iHeight * layers[0]->iDepth;
        }

        size_t getOutSize() const {
            size_t last = layers.size() - 1;
            return layers[last]->out.size();
        }

        const vec &getOut() const {
            return layers[layers.size() - 1]->out;
        }

    private:

        void initOpenCL(const std::string &kernelFileName, bool isBinary, size_t inSize) {
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

            if (isBinary) {
                program = buildProgramFromBinary(kernelFileName.c_str(), context, device);
            }
            else {
                program = buildProgramFromSource(kernelFileName.c_str(), context, device);
            }
            err = clRetainProgram(program);
            handleError(err, "Failed retaining program. ");

            clIn = clCreateBuffer(
                context,
                CL_MEM_READ_ONLY,
                inSize * sizeof(cl_float),
                NULL,
                &err);
            handleError(err, "Failed creating clIn");
            err = clRetainMemObject(clIn);
            handleError(err, "Failed retaining clIn");
        }

        // Create a layer.
        Layer *createLayer(rapidxml::xml_node<> *root, Flag flag) {

            std::string type = getString(root, "type");
            if (type == "conv") {
                return createConvLayer(root, flag);
            }
            else if (type == "max") {
                return createMaxLayer(root, flag);
            }
            else {
                std::cerr << "createLayer: Unsupported layer: " << type << std::endl;
                exit(-1);
            }
        }

        Layer *createMaxLayer(rapidxml::xml_node<> *root, Flag flag) {
            LayerParam params;

            params.flag = flag;

            // Get the parameters for the convolutional layer.
            params.iWidth = getSizeT(root, "iWidth");
            params.iHeight = getSizeT(root, "iHeight");
            params.iDepth = getSizeT(root, "iDepth");
            params.kernelSize = getSizeT(root, "kernelSize");
            params.oDepth = params.iDepth;
            params.oWidth = params.iWidth / params.kernelSize;
            params.oHeight = params.iHeight / params.kernelSize;

            // Get the kernel name.
            params.kernelName = getString(root, "kernelName");

            // Get the work group size.
            std::vector<size_t> workGroupSize;
            getAllItem(root->first_node("workGroupSize"), workGroupSize);
            for (size_t i = 0; i < workGroupSize.size(); ++i) {
                params.workGroupSize[i] = workGroupSize[i];
            }

            return new cnn::MaxPoolLayer(params,
                context,
                program,
                clIn
                );
        }

        Layer *createConvLayer(rapidxml::xml_node<> *root, Flag flag) {

            LayerParam params;

            params.flag = flag;

            // Get the parameters for the convolutional layer.
            params.iWidth       = getSizeT(root, "iWidth");
            params.iHeight      = getSizeT(root, "iHeight");
            params.iDepth       = getSizeT(root, "iDepth");
            params.kernelSize   = getSizeT(root, "kernelSize");
            params.oDepth       = getSizeT(root, "oDepth");

            params.oWidth       = params.iWidth - params.kernelSize + 1;
            params.oHeight      = params.iHeight - params.kernelSize + 1;

            // Get the kernel name.
            params.kernelName   = getString(root, "kernelName");

            // Get the work group size.
            std::vector<size_t> workGroupSize;
            getAllItem(root->first_node("workGroupSize"), workGroupSize);
            for (size_t i = 0; i < workGroupSize.size(); ++i) {
                params.workGroupSize[i] = workGroupSize[i];
            }

            // Create the weight vector.
            cnn::vec weight;
            getAllItem(root->first_node("weight"), weight);
            assert(weight.size() == params.oDepth * params.iDepth * params.kernelSize * params.kernelSize);

            // Create the offset vector.
            cnn::vec offset;
            for (rapidxml::xml_node<> *node = root->first_node("offset")->first_node(); node; node = node->next_sibling()) {
                offset.push_back((float)std::atof(node->value()));
            }
            assert(offset.size() == params.oDepth);

            return new cnn::ConvolutionLayer(params,
                weight,
                offset,
                context,
                program,
                clIn
                );
        }

    };
}



#endif