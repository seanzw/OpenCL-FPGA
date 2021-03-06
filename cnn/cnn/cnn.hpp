#ifndef CNN_HEADER
#define CNN_HEADER

#include <map>

#include "util.hpp"
#include "convolution.hpp"
#include "maxpool.hpp"
#include "fullconnect.hpp"
#include "rbf.hpp"
#include "eventpool.hpp"


#define BUFSIZE (64 * 1024 * 1024)

namespace cnn {
    class CNN {
    public:

        CNN(const std::string &xmlFileName, bool isQueueInOrder = true, const std::string &xclbinFile = "NONE") {

            this->isQueueInOrder = isQueueInOrder;

            // Parse the xml file.
            char *buf = new char[BUFSIZE];
            fileToChar(xmlFileName, buf, BUFSIZE);
            
            rapidxml::xml_document<> doc;
            doc.parse<0>(buf);
            rapidxml::xml_node<> *root = doc.first_node();

            // Get the input size.
            size_t inSize = getSizeT(root, "inSize");

            // Get the queue barrier.
            queueBarrier = getSizeT(root, "queueBarrier");

            // Initialize the OpenCL.
            initOpenCL(isQueueInOrder, inSize);

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
                layers.push_back(createLayer(layer, xclbinFile != "NONE", flag));
            }

            delete[] buf;
        }

        ~CNN() {
            for (int i = 0; i < layers.size(); ++i) {
                delete layers[i];
            }
            for (std::map<std::string, cl_program>::iterator iter = programs.begin(); iter != programs.end(); ++iter) {
                clReleaseProgram(iter->second);
            }
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

        unsigned long long forwardCPUBatch(const vec &in, vec &out, size_t n, double *averageTime) {
            // Make sure that input size is correct.
            size_t inSize = getInSize();
            size_t outSize = getOutSize();
            if (in.size() != inSize * n) {
                std::cerr << "Wrong input size! " << std::endl;
                exit(-2);
            }

            clock_t start = clock(), diff;

            // Reserve the output buffer.
            out.resize(outSize * n);

            for (size_t i = 0; i < n; ++i) {
                vec inThis(in.begin() + i * inSize, in.begin() + i * inSize + inSize);
                forwardCPU(inThis);
                std::copy(layers[layers.size() - 1]->out.begin(), layers[layers.size() - 1]->out.end(), out.begin() + i * outSize);
            }

            diff = clock() - start;
            *averageTime = (double)diff / (double)CLOCKS_PER_SEC / (double)n;
            std::cout << "Average time (CPU): " << *averageTime << "s" << std::endl;

            return diff;
        }

        // Forward with OpenCL.
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

        // Forward more than one input with in order command queue.
        std::vector<cl_event> forwardCLBatch(const vec &in, vec &out, size_t n, double *averageTime) {

            // Make sure that input size is correct.
            size_t inSize = getInSize();
            size_t outSize = getOutSize();
            size_t eventSize = layers.size() + 2;
            if (in.size() != inSize * n) {
                std::cerr << "Wrong input size! " << std::endl;
                exit(-2);
            }

            clock_t start = clock(), diff;

            // Reserve the output buffer.
            out.resize(outSize * n);

            // Reserve the event buffer.
            // One event for each layer plus two events for IO.
            std::vector<cl_event> events(n * eventSize);

            // For OpenCL error.
            cl_int err;

            for (size_t i = 0; i < n; ++i) {
                
                // Prepare the input cl_mem.
                err = clEnqueueWriteBuffer(queue,
                    clIn,
                    CL_FALSE,
                    0,
                    inSize * sizeof(cl_float),
                    (void *)&in[i * inSize],
                    i == 0 ? 0 : 1,
                    i == 0 ? NULL : &events[(i - 1) * eventSize],
                    &events[i * eventSize]);
                handleError(err, "Failed copy input buffer. ");

                // For each layer.
                for (size_t l = 0; l < layers.size(); ++l) {
                    err = clEnqueueNDRangeKernel(queue,
                        layers[l]->kernel,
                        3,
                        NULL,
                        layers[l]->global,
                        layers[l]->workGroupSize,
                        1,
                        &events[i * eventSize + l],
                        &events[i * eventSize + l + 1]);
                    handleError(err, "Failed enqueuing kernel. ");
                }

                // Get the output.
                err = clEnqueueReadBuffer(queue,
                    layers[layers.size() - 1]->clOut,
                    CL_FALSE,
                    0,
                    outSize * sizeof(cl_float),
                    &out[i * outSize],
                    1,
                    &events[i * eventSize + layers.size()],
                    &events[i * eventSize + layers.size() + 1]);
                handleError(err, "Failed enqueuing reading buffer. ");

                // Wait for the command queue.
                if (i % queueBarrier == queueBarrier - 1) {
                    err = clFinish(queue);
                    handleError(err, "Failed waiting for event. ");
                }

            }

            diff = clock() - start;
            *averageTime = (double)diff / (double)CLOCKS_PER_SEC / (double)n;
            std::cout << "Average time: " << *averageTime << "s" << std::endl;

            return events;
        }

        // Forward with pipelined command queue.
        std::vector<cl_event> forwardCLPipeline(const vec &in, vec &out, size_t n, double *averageTime) {

            // Check if the command queue supports out of order queue.
            if (isQueueInOrder) {
                std::cout << "Warning: using an in order command queue for pipeline cnn!" << std::endl;
            }

            // Make sure that input size is correct.
            size_t inSize = getInSize();
            size_t outSize = getOutSize();
            size_t eventSize = layers.size() + 2;
            if (in.size() != inSize * n) {
                std::cerr << "Wrong input size! " << std::endl;
                exit(-2);
            }

            // Initialize the event pool.
            EventPool events(layers.size() + 2, n);

            clock_t start = clock(), diff;

            // Reserve the output buffer.
            out.resize(outSize * n);

            // For OpenCL error.
            cl_int err;

            // Temporary holder for returned event.
            cl_event event;
            uint32_t len;
            cl_event *eventList;

            for (size_t i = 0; i < n; ++i) {

                // Prepare the input cl_mem.
                eventList = events.getDependentEventList(0, i, &len);
                err = clEnqueueWriteBuffer(queue,
                    clIn,
                    CL_FALSE,
                    0,
                    inSize * sizeof(cl_float),
                    (void *)&in[i * inSize],
                    len,
                    eventList,
                    &event);
                handleError(err, "Failed copy input buffer. ");
                events.pushEvent(0, i, event);

                // For each layer.
                for (size_t l = 0; l < layers.size(); ++l) {
                    eventList = events.getDependentEventList(l + 1, i, &len);
                    err = clEnqueueNDRangeKernel(queue,
                        layers[l]->kernel,
                        3,
                        NULL,
                        layers[l]->global,
                        layers[l]->workGroupSize,
                        len,
                        eventList,
                        &event);
                    handleError(err, "Failed enqueuing kernel. ");
                    events.pushEvent(l + 1, i, event);
                }

                // Get the output.
                eventList = events.getDependentEventList(layers.size() + 1, i, &len);
                err = clEnqueueReadBuffer(queue,
                    layers[layers.size() - 1]->clOut,
                    CL_FALSE,
                    0,
                    outSize * sizeof(cl_float),
                    &out[i * outSize],
                    len,
                    eventList,
                    &event);
                handleError(err, "Failed enqueuing reading buffer. ");
                events.pushEvent(layers.size() + 1, i, event);

                // Wait for the command queue.
                if (i % queueBarrier == queueBarrier - 1) {
                    err = clFinish(queue);
                    handleError(err, "Failed waiting for event. ");
                }
            }

            diff = clock() - start;
            *averageTime = (double)diff / (double)CLOCKS_PER_SEC / (double)n;
            std::cout << "Pipelined average time: " << *averageTime << "s" << std::endl;

            return events.sort();
        }

        // For OpenCL.
        cl_platform_id platform;
        cl_device_id device;
        cl_context context;
        cl_command_queue queue;
        std::map<std::string, cl_program> programs;
        cl_mem clIn;

        size_t queueBarrier;
        bool isQueueInOrder;

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

        void initOpenCL(bool isQueueInOrder, size_t inSize) {
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
                CL_QUEUE_PROFILING_ENABLE | (isQueueInOrder ? 0 : CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE),
                &err);
            handleError(err, "Failed creating command queue. ");
            clRetainCommandQueue(queue);

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
        Layer *createLayer(rapidxml::xml_node<> *root, bool isBinary, Flag flag) {

            LayerParam params;

            params.flag = flag;

            // Get the parameters for the convolutional layer.
            params.iWidth = getSizeT(root, "iWidth");
            params.iHeight = getSizeT(root, "iHeight");
            params.iDepth = getSizeT(root, "iDepth");
            params.oWidth = getSizeT(root, "oWidth");
            params.oHeight = getSizeT(root, "oHeight");
            params.oDepth = getSizeT(root, "oDepth");
            params.oWidthTile = getSizeT(root, "oWidthTile");
            params.oHeightTile = getSizeT(root, "oHeightTile");
            params.oDepthTile = getSizeT(root, "oDepthTile");
            params.iDepthTile = getSizeT(root, "iDepthTile");
            params.kernelSize = getSizeT(root, "kernelSize");

            // Get the kernel name.
            params.kernelName = getString(root, "kernelName");

            // Get the work group size.
            std::vector<size_t> workGroupSize;
            getAllItem(root->first_node("workGroupSize"), workGroupSize);
            for (size_t i = 0; i < workGroupSize.size(); ++i) {
                params.workGroupSize[i] = workGroupSize[i];
            }

            // Create the weight vector.
            cnn::vec weight;
            getAllItem(root->first_node("weight"), weight);

            // Create the offset vector.
            cnn::vec offset;
            getAllItem(root->first_node("offset"), offset);

            // Get the program.
            cl_program program;
            if (isBinary) {
                std::string xclbinFileName = getString(root, "xclbinFileName");
                std::map<std::string, cl_program>::iterator iter = programs.find(xclbinFileName);
                if (iter != programs.end()) {
                    program = iter->second;
                }
                else {
                    program = buildProgramFromBinary(xclbinFileName.c_str(), context, device);
                    cl_int err = clRetainProgram(program);
                    handleError(err, "Failed retaining program. ");
                    programs.insert(std::pair<std::string, cl_program>(xclbinFileName, program));
                }
            }
            else {
                std::string kernelFileName = getString(root, "kernelFileName");
                std::map<std::string, cl_program>::iterator iter = programs.find(kernelFileName);
                if (iter != programs.end()) {
                    program = iter->second;
                }
                else {
                    program = buildProgramFromSource(kernelFileName.c_str(), context, device);
                    cl_int err = clRetainProgram(program);
                    handleError(err, "Failed retaining program. ");
                    programs.insert(std::pair<std::string, cl_program>(kernelFileName, program));
                }
            }

            std::string type = getString(root, "type");
            if (type == "conv") {
                return new cnn::ConvolutionLayer(params,
                    weight,
                    offset,
                    context,
                    program,
                    clIn
                    );
            }
            else if (type == "pool") {
                return new cnn::MaxPoolLayer(params,
                    weight,
                    offset,
                    context,
                    program,
                    clIn
                    );
            }
            else if (type == "full") {
                return new cnn::FullConnectLayer(params,
                    weight,
                    offset,
                    context,
                    program,
                    clIn
                    );
            }
            else if (type == "rbf") {
                return new cnn::RBFLayer(params,
                    weight,
                    offset,
                    context,
                    program,
                    clIn
                    );
            }
            else {
                std::cerr << "createLayer: Unsupported layer: " << type << std::endl;
                exit(-1);
            }
        }

    };
}

#endif
