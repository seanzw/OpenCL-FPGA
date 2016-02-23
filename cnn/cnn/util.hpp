#ifndef UTIL_HPP
#define UTIL_HPP

#include <exception>
#include <fstream>
#include <sstream>
#include <iostream>
#include <vector>

#define __CL_ENABLE_EXCEPTIONS
#include <CL/cl.h>
#include "include/RapidXML/rapidxml.hpp"

namespace cnn {

    const char *readable_status(cl_int status) {
        switch (status) {
        case CL_SUCCESS:
            return "CL_SUCCESS";
        case CL_DEVICE_NOT_FOUND:
            return "CL_DEVICE_NOT_FOUND";
        case CL_DEVICE_NOT_AVAILABLE:
            return "CL_DEVICE_NOT_AVAILABLE";
        case CL_COMPILER_NOT_AVAILABLE:
            return "CL_COMPILER_NOT_AVAILABLE";
        case CL_MEM_OBJECT_ALLOCATION_FAILURE:
            return "CL_COMPILER_NOT_AVAILABLE";
        case CL_OUT_OF_RESOURCES:
            return "CL_OUT_OF_RESOURCES";
        case CL_OUT_OF_HOST_MEMORY:
            return "CL_OUT_OF_HOST_MEMORY";
        case CL_PROFILING_INFO_NOT_AVAILABLE:
            return "CL_PROFILING_INFO_NOT_AVAILABLE";
        case CL_MEM_COPY_OVERLAP:
            return "CL_MEM_COPY_OVERLAP";
        case CL_IMAGE_FORMAT_MISMATCH:
            return "CL_IMAGE_FORMAT_MISMATCH";
        case CL_IMAGE_FORMAT_NOT_SUPPORTED:
            return "CL_IMAGE_FORMAT_NOT_SUPPORTED";
        case CL_BUILD_PROGRAM_FAILURE:
            return "CL_BUILD_PROGRAM_FAILURE";
        case CL_MAP_FAILURE:
            return "CL_MAP_FAILURE";
#ifndef CL_VERSION_1_0
        case CL_MISALIGNED_SUB_BUFFER_OFFSET:
            return "CL_MISALIGNED_SUB_BUFFER_OFFSET";
        case CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST:
            return "CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST";
#endif
        case CL_INVALID_VALUE:
            return "CL_INVALID_VALUE";
        case CL_INVALID_DEVICE_TYPE:
            return "CL_INVALID_DEVICE_TYPE";
        case CL_INVALID_PLATFORM:
            return "CL_INVALID_PLATFORM";
        case CL_INVALID_DEVICE:
            return "CL_INVALID_DEVICE";
        case CL_INVALID_CONTEXT:
            return "CL_INVALID_CONTEXT";
        case CL_INVALID_QUEUE_PROPERTIES:
            return "CL_INVALID_QUEUE_PROPERTIES";
        case CL_INVALID_COMMAND_QUEUE:
            return "CL_INVALID_COMMAND_QUEUE";
        case CL_INVALID_HOST_PTR:
            return "CL_INVALID_HOST_PTR";
        case CL_INVALID_MEM_OBJECT:
            return "CL_INVALID_MEM_OBJECT";
        case CL_INVALID_IMAGE_FORMAT_DESCRIPTOR:
            return "CL_INVALID_IMAGE_FORMAT_DESCRIPTOR";
        case CL_INVALID_IMAGE_SIZE:
            return "CL_INVALID_IMAGE_SIZE";
        case CL_INVALID_SAMPLER:
            return "CL_INVALID_SAMPLER";
        case CL_INVALID_BINARY:
            return "CL_INVALID_BINARY";
        case CL_INVALID_BUILD_OPTIONS:
            return "CL_INVALID_BUILD_OPTIONS";
        case CL_INVALID_PROGRAM:
            return "CL_INVALID_PROGRAM";
        case CL_INVALID_PROGRAM_EXECUTABLE:
            return "CL_INVALID_PROGRAM_EXECUTABLE";
        case CL_INVALID_KERNEL_NAME:
            return "CL_INVALID_KERNEL_NAME";
        case CL_INVALID_KERNEL_DEFINITION:
            return "CL_INVALID_KERNEL_DEFINITION";
        case CL_INVALID_KERNEL:
            return "CL_INVALID_KERNEL";
        case CL_INVALID_ARG_INDEX:
            return "CL_INVALID_ARG_INDEX";
        case CL_INVALID_ARG_VALUE:
            return "CL_INVALID_ARG_VALUE";
        case CL_INVALID_ARG_SIZE:
            return "CL_INVALID_ARG_SIZE";
        case CL_INVALID_KERNEL_ARGS:
            return "CL_INVALID_KERNEL_ARGS";
        case CL_INVALID_WORK_DIMENSION:
            return "CL_INVALID_WORK_DIMENSION";
        case CL_INVALID_WORK_GROUP_SIZE:
            return "CL_INVALID_WORK_GROUP_SIZE";
        case CL_INVALID_WORK_ITEM_SIZE:
            return "CL_INVALID_WORK_ITEM_SIZE";
        case CL_INVALID_GLOBAL_OFFSET:
            return "CL_INVALID_GLOBAL_OFFSET";
        case CL_INVALID_EVENT_WAIT_LIST:
            return "CL_INVALID_EVENT_WAIT_LIST";
        case CL_INVALID_EVENT:
            return "CL_INVALID_EVENT";
        case CL_INVALID_OPERATION:
            return "CL_INVALID_OPERATION";
        case CL_INVALID_GL_OBJECT:
            return "CL_INVALID_GL_OBJECT";
        case CL_INVALID_BUFFER_SIZE:
            return "CL_INVALID_BUFFER_SIZE";
        case CL_INVALID_MIP_LEVEL:
            return "CL_INVALID_MIP_LEVEL";
        case CL_INVALID_GLOBAL_WORK_SIZE:
            return "CL_INVALID_GLOBAL_WORK_SIZE";
#ifndef CL_VERSION_1_0
        case CL_INVALID_PROPERTY:
            return "CL_INVALID_PROPERTY";
#endif
        default:
            return "CL_UNKNOWN_CODE";
        }
    }

    std::string fileToString(const std::string &fn) {
        std::string text;
        std::ifstream fs(fn.c_str());
        if (!fs) {
            std::ostringstream os;
            os << "There is no file called " << fn;
            exit(-1);
        }
        text.assign(std::istreambuf_iterator<char>(fs), std::istreambuf_iterator<char>());
        return text;
    }

    cl_program buildProgramFromSource(const std::string &fileName, const cl_context &context, const cl_device_id &device) {
        std::string text = fileToString(fileName);
        const char *source = text.c_str();
        cl_int err;
        cl_program program = clCreateProgramWithSource(context, 1, (const char **)&source, NULL, NULL);
        if (program == NULL) {
            std::cerr << "Failed to create CL program from source. " << std::endl;
            exit(-1);
        }
        
        err = clBuildProgram(program, 0, NULL, NULL, NULL, NULL);
        if (err != CL_SUCCESS) {
            char buildLog[16384];
            clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, sizeof(buildLog), buildLog, NULL);
            std::cerr << "Error in kernel: " << std::endl;
            std::cerr << buildLog;
            clReleaseProgram(program);
            exit(-1);
        }

        return program;
    }

    cl_program buildProgramFromBinary(const std::string &fileName, const cl_context &context, const cl_device_id &device) {
        std::string text = fileToString(fileName);
        size_t len = text.size();
        const char *source = text.c_str();
        cl_int err;
        cl_program program = clCreateProgramWithBinary(context,
            1,
            &device,
            &len,
            (const unsigned char **)&source,
            NULL,
            &err);
        if (program == NULL || err != CL_SUCCESS) {
            std::cerr << "Failed to create CL program from source. " << std::endl;
            exit(-1);
        }

        err = clBuildProgram(program, 0, NULL, NULL, NULL, NULL);
        if (err != CL_SUCCESS) {
            char buildLog[16384];
            clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, sizeof(buildLog), buildLog, NULL);
            std::cerr << "Error in kernel: " << std::endl;
            std::cerr << buildLog;
            clReleaseProgram(program);
            exit(-1);
        }

        return program;
    }

    cl_ulong runAndTimeKernel(const cl_command_queue &queue,
        const cl_kernel &kernel,
        const cl_uint workDim,
        const size_t *globalSize,
        const size_t *localSize
        ) {
        cl_event event;
        cl_ulong t1, t2;
        cl_int err = clEnqueueNDRangeKernel(queue,
            kernel,
            workDim,
            NULL,
            globalSize,
            localSize,
            0,
            NULL,
            &event);
        if (err != CL_SUCCESS) {
            std::cerr << "Error enqueuing kernel for execution. " << std::endl;
            std::cerr << readable_status(err);
            exit(-1);
        }

        clWaitForEvents(1, &event);

        err = clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &t1, NULL);
        err = clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &t2, NULL);
        if (err != CL_SUCCESS) {
            std::cerr << "Error timing the kernel. " << std::endl;
            std::cerr << readable_status(err);
            exit(-1);
        }

        return t2 - t1;
    }

   

    unsigned int closestMultiple(unsigned int size, unsigned int divisor) {
        unsigned int remainder = size % divisor;
        return remainder == 0 ? size : size - remainder + divisor;
    }

    template <class T>
    void showMatrix(T *matrix, unsigned int width, unsigned int height) {
        for (unsigned int row = 0; row < height; ++row) {
            for (unsigned int col = 0; col < width; ++col) {
                std::cout << matrix[width*row + col] << " ";
            }
            std::cout << std::endl;
        }
        return;
    }

    typedef std::vector<float> vec;
    typedef std::vector<vec> vec2d;

    int getInt(rapidxml::xml_node<> *root, const char *name) {
        rapidxml::xml_node<> *node = root->first_node(name);
        return std::atoi(node->value());
    }

    void getAllItem(rapidxml::xml_node<> *root, vec &items) {
        std::string name = root->name();
        if (name == "item") {
            items.push_back((float)std::atof(root->value()));
        }
        else {
            for (rapidxml::xml_node<> *node = root->first_node(); node; node = node->next_sibling()) {
                getAllItem(node, items);
            }
        }
        return;
    }

    int closestMultiple(int base, int n) {
        int remainder = n % base;
        return remainder == 0 ? n : n - remainder + base;
    }
}


#endif
