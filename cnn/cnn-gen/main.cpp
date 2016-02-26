#include <random>
#include <iostream>
#include <fstream>
#include <sstream>
#include <cstdio>
#include "CL/cl.h"

class CNNGenerator {
public:
    enum LayerType {
        CONV,
        SUB,
        FULL
    };

    struct LayerParam {
        LayerType type;
        std::string kernelName;
        size_t workGroupSize[3];
        size_t iWidth;
        size_t iHeight;
        size_t iDepth;
        size_t kernelSize;
        size_t oDepth;
    };

    static void genCNN(const std::string &XMLFileName,
        const std::string &kernelFileName,
        size_t layerNum,
        const LayerParam *layerParams
        ) {

        std::ofstream xml(XMLFileName);
        if (!xml.is_open()) {
            std::cerr << "Can't open file " << XMLFileName << std::endl;
            exit(-1);
        }

        FILE *kernel;
        fopen_s(&kernel, kernelFileName.c_str(), "w");
        if (kernel == NULL) {
            std::cerr << "Can't open file " << kernelFileName << std::endl;
            exit(-1);
        }

        // Write some basic information in the xml file.
        xml << "<?xml version=\"1.0\" encoding=\"utf-8\"?>" << std::endl;
        writeXMLOpenTag(xml, "cnn");
        writeXMLTag(xml, "kernelFileName", kernelFileName);

        // Write some basic information in the cl kernel file.
        fprintf(kernel, "%s\n", activateFunc.c_str());

        for (size_t i = 0; i < layerNum; ++i) {
            genLayer(xml, kernel, layerParams[i]);
        }

        writeXMLCloseTag(xml, "cnn");

        xml.close();
        fclose(kernel);
    }

private:

    static void genLayer(std::ofstream &xml, FILE *kernel, const LayerParam &param) {
        writeXMLOpenTag(xml, "layer");
        switch (param.type) {
        case CONV:
            genXMLConvLayer(xml, param);
            genCLConvLayer(kernel, param);
            break;
        default:
            std::cerr << "Unsupported layer type. " << std::endl;
            exit(-1);
        }
        writeXMLCloseTag(xml, "layer");
    }

    // Generate the xml for this layer.
    static void genXMLConvLayer(std::ofstream &xml, const LayerParam &param) {
        writeXMLTag(xml, "kernelName", param.kernelName);

        writeXMLOpenTag(xml, "workGroupSize");
        for (size_t i = 0; i < sizeof(param.workGroupSize) / sizeof(size_t); ++i) {
            writeXMLTag(xml, "item", param.workGroupSize[i]);
        }
        writeXMLCloseTag(xml,"workGroupSize");

        writeXMLTag(xml, "iWidth", param.iWidth);
        writeXMLTag(xml, "iHeight", param.iHeight);
        writeXMLTag(xml, "iDepth", param.iDepth);
        writeXMLTag(xml, "kernelSize", param.kernelSize);
        writeXMLTag(xml, "oDepth", param.oDepth);

        // Randomly write the weight.
        writeXMLOpenTag(xml, "weight");
        for (int i = 0; i < param.oDepth; ++i) {
            // For each output feature map.
            writeXMLOpenTag(xml, "oFeatureMap");
            for (int j = 0; j < param.iDepth; ++j) {
                writeXMLOpenTag(xml, "iFeatureMap");
                for (int k = 0; k < param.kernelSize; ++k) {
                    writeXMLOpenTag(xml, "line");
                    for (int k = 0; k < param.kernelSize; ++k) {
                        writeXMLTag(xml, "item", static_cast<float>(rand()) / static_cast<float>(RAND_MAX));
                    }
                    writeXMLCloseTag(xml, "line");
                }
                writeXMLCloseTag(xml, "iFeatureMap");
            }
            writeXMLCloseTag(xml, "oFeatureMap");
        }
        writeXMLCloseTag(xml, "weight");

        // Randomly write the offset.
        writeXMLOpenTag(xml, "offset");
        for (int i = 0; i < param.oDepth; ++i) {
            writeXMLTag(xml, "item", static_cast<float>(rand()) / static_cast<float>(RAND_MAX));
        }
        writeXMLCloseTag(xml, "offset");
    }

    // Generate the OpenCL kernel for convolution layer.
    static void genCLConvLayer(FILE *kernel, const LayerParam &param) {

        writeDefine(kernel, "KERNEL_SIZE", param.kernelSize);
        writeDefine(kernel, "KERNEL_LEN", param.kernelSize * param.kernelSize);
        writeDefine(kernel, "IWIDTH", param.iWidth);
        writeDefine(kernel, "IHEIGHT", param.iHeight);
        writeDefine(kernel, "IDEPTH", param.iDepth);
        writeDefine(kernel, "OWIDTH", param.iWidth - param.kernelSize + 1);
        writeDefine(kernel, "OHEIGHT", param.iHeight - param.kernelSize + 1);
        writeDefine(kernel, "ODEPTH", param.oDepth);

        fprintf(kernel, convKernel.c_str(),
            (int)param.workGroupSize[0],
            (int)param.workGroupSize[1],
            (int)param.workGroupSize[2],
            param.kernelName.c_str());

        writeUndef(kernel, "KERNEL_SIZE");
        writeUndef(kernel, "KERNEL_LEN");
        writeUndef(kernel, "IWIDTH");
        writeUndef(kernel, "IHEIGHT");
        writeUndef(kernel, "IDEPTH");
        writeUndef(kernel, "OWIDTH");
        writeUndef(kernel, "OHEIGHT");
        writeUndef(kernel, "ODEPTH");
    }

    /***********************************************************************
     Helper function to generate the xml tags.
     ************************************************************************/
    static void writeXMLOpenTag(std::ofstream &o, const std::string &tag) {
        o << "<" << tag << ">";
    }

    static void writeXMLCloseTag(std::ofstream &o, const std::string &tag) {
        o << "</" << tag << ">";
    }

    static void writeXMLTag(std::ofstream &o, const std::string &tag, float value) {
        writeXMLOpenTag(o, tag);
        o << value;
        writeXMLCloseTag(o, tag);
        o << std::endl;
    }

    static void writeXMLTag(std::ofstream &o, const std::string &tag, size_t value) {
        writeXMLOpenTag(o, tag);
        o << value;
        writeXMLCloseTag(o, tag);
        o << std::endl;
    }

    static void writeXMLTag(std::ofstream &o, const std::string &tag, const std::string &value) {
        writeXMLOpenTag(o, tag);
        o << value;
        writeXMLCloseTag(o, tag);
        o << std::endl;
    }

    /***********************************************************************
     Helper function to generate OpenCL kernel file.
    ************************************************************************/
    static void writeDefine(FILE *o, const std::string &macro, size_t value) {
        fprintf(o, "#define %s %u\n", macro.c_str(), (unsigned int)value);
    }

    static void writeUndef(FILE *o, const std::string &macro) {
        fprintf(o, "#undef %s\n", macro.c_str());
    }

    /********************************************************************************************
     Some constant value.
     ********************************************************************************************/
    static const std::string activateFunc;
    static const std::string convKernel;
};

/* Initialize the constant value. */
const std::string CNNGenerator::activateFunc = "\
float sigmod(float in) {\n\
    return 1.0f / (1.0f + exp(-in)); \n\
}";

const std::string CNNGenerator::convKernel = "\
#ifdef __xilinx__\n\
__attribute__ ((reqd_work_group_size(%d, %d, %d)))\n\
#endif\n\
__kernel void %s(\n\
    __global float *in,\n\
    __constant float *weight,\n\
    __constant float *offset,\n\
    __global float *out\n\
    ) {\n\
    \n\
    #ifdef __xilinx__\n\
    __attribute__((xcl_pipeline_workitems))\n\
    #endif\n\
    int c = get_global_id(0);\n\
    int r = get_global_id(1);\n\
    \n\
    if (c < OWIDTH && r < ODEPTH * OHEIGHT) {\n\
\n\
        // Get the index of the element in output feature map.\n\
        int o = r / OHEIGHT;\n\
        r = r %% OHEIGHT; \n\
\n\
        float sum = 0.0f;\n\
\n\
        // For each input feature map.\n\
        #ifdef __xilinx__\n\
        __attribute__((xcl_pipeline_loop))\n\
        #endif\n\
        for (int i = 0; i < IDEPTH; ++i) {\n\
            \n\
            float inputBuf[KERNEL_LEN];\n\
            float weightBuf[KERNEL_LEN];\n\
            int idx = 0;\n\
            int weightBase = (o * IDEPTH + i) * KERNEL_LEN;\n\
            #ifdef __xilinx__\n\
            __attribute__((opencl_unroll_hint))\n\
            #endif\n\
            for (int x = 0; x < KERNEL_SIZE; ++x) {\n\
                #ifdef __xilinx__\n\
                __attribute__((opencl_unroll_hint))\n\
                #endif\n\
                for (int y = 0; y < KERNEL_SIZE; ++y) {\n\
                    inputBuf[idx] = in[(i * IHEIGHT + r + x) * IWIDTH + c + y];\n\
                    weightBuf[idx] = weight[weightBase + idx];\n\
                    idx++;\n\
                }\n\
            }\n\
\n\
            #ifdef __xilinx__\n\
            __attribute__((opencl_unroll_hint))\n\
            #endif\n\
            for (int x = 0; x < KERNEL_LEN; ++x) {\n\
                sum += inputBuf[x] * weightBuf[x];\n\
            }\n\
        }\n\
\n\
        // Get the output index.\n\
        int outIdx = (o * OHEIGHT + r) * OWIDTH + c;\n\
        out[outIdx] = sigmod(sum + offset[o]);\n\
    } \n\
}\n";


int main(int argc, char *argv[]) {

    CNNGenerator::LayerParam params[] = {
        {
            CNNGenerator::CONV,
            "conv1",
            {16, 1, 1},
            32,
            32,
            1,
            5,
            6
        }
    };

    CNNGenerator::genCNN("../cnn/conv1.xml", "../cnn/conv1.cl", 1, params);

    return 0;
}