#include <random>
#include <iostream>
#include <fstream>
#include <sstream>
#include <cstdio>

class CNNGenerator {
public:
    enum LayerType {
        CONV,
        POOL,
        FULL,
        RBF
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
        size_t oWidthTile;
        size_t oHeightTile;
        size_t oDepthTile;
        size_t iDepthTile;
    };

    static void genCNN(const std::string &XMLFileName,
        const std::string &kernelFileName,
        size_t layerNum,
        const LayerParam *params
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
        writeXMLTag(xml, "inSize", params[0].iWidth * params[0].iHeight * params[0].iDepth);
        writeXMLTag(xml, "queueBarrier", static_cast<size_t>(10));

        // Write some basic information in the cl kernel file.
        fprintf(kernel, "%s\n", activateFunc.c_str());

        // Write the internal buffer.
        for (size_t i = 1; i < layerNum; ++i) {
            fprintf(kernel,
                "__global float buf%zu[%zu];\n",
                i,
                params[i].iWidth * params[i].iHeight * params[i].iDepth);
        }

        for (size_t i = 0; i < layerNum; ++i) {
            Flag flag = INNER;
            if (i == 0) {
                flag |= FRONT;
            }
            if (i == layerNum - 1) {
                flag |= BACK;
            }
            genLayer(xml, kernel, params[i], i, flag);
        }

        writeXMLCloseTag(xml, "cnn");

        xml.close();
        fclose(kernel);
    }

private:

    static void genLayer(std::ofstream &xml, FILE *kernel, const LayerParam &param, size_t idx, Flag flag) {
        writeXMLOpenTag(xml, "layer");
        writeKernelDefine(kernel, param, idx, flag);
        switch (param.type) {
        case CONV:
            genXMLConvLayer(xml, param);
            fprintf(kernel, "%s\n", convKernel.c_str());
            break;
        case POOL:
            genXMLPoolLayer(xml, param);
            fprintf(kernel, "%s\n", poolKernel.c_str());
            break;
        case FULL:
            genXMLFullLayer(xml, param);
            fprintf(kernel, "%s\n", fullKernel.c_str());
            break;
        case RBF:
            genXMLRBFLayer(xml, param);
            fprintf(kernel, "%s\n", rbfKernel.c_str());
            break;
        default:
            std::cerr << "Unsupported layer type. " << std::endl;
            exit(-1);
        }
        writeKernelUndefine(kernel, flag);
        writeXMLCloseTag(xml, "layer");
    }

    static void genXMLPoolLayer(std::ofstream &xml, const LayerParam &param) {
        writeXMLTag(xml, "type", "pool");
        writeXMLInfo(xml, param);

        // Randomly write the weight.
        writeXMLOpenTag(xml, "weight");
        for (int i = 0; i < param.oDepth; ++i) {
            writeXMLTag(xml, "item", static_cast<float>(rand()) / static_cast<float>(RAND_MAX));
        }
        writeXMLCloseTag(xml, "weight");

        // Randomly write the offset.
        writeXMLOpenTag(xml, "offset");
        for (int i = 0; i < param.oDepth; ++i) {
            writeXMLTag(xml, "item", static_cast<float>(rand()) / static_cast<float>(RAND_MAX));
        }
        writeXMLCloseTag(xml, "offset");
    }

    static void genXMLFullLayer(std::ofstream &xml, const LayerParam &param) {
        writeXMLTag(xml, "type", "full");
        writeXMLInfo(xml, param);

        // Randomly write the weight.
        writeXMLOpenTag(xml, "weight");
        for (int i = 0; i < param.oWidth * param.oHeight * param.oDepth; ++i) {
            // For each output feature map.
            for (int k = 0; k < param.iWidth * param.iHeight * param.iDepth; ++k) {
                writeXMLTag(xml, "item", static_cast<float>(rand()) / static_cast<float>(RAND_MAX));
            }
        }
        writeXMLCloseTag(xml, "weight");

        // Randomly write the offset.
        writeXMLOpenTag(xml, "offset");
        for (int i = 0; i < param.oWidth * param.oHeight * param.oDepth; ++i) {
            writeXMLTag(xml, "item", static_cast<float>(rand()) / static_cast<float>(RAND_MAX));
        }
        writeXMLCloseTag(xml, "offset");
    }

    static void genXMLConvLayer(std::ofstream &xml, const LayerParam &param) {
        writeXMLTag(xml, "type", "conv");
        writeXMLInfo(xml, param);

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

    static void genXMLRBFLayer(std::ofstream &xml, const LayerParam &param) {
        writeXMLTag(xml, "type", "rbf");
        writeXMLInfo(xml, param);

        // Randomly write the weight.
        writeXMLOpenTag(xml, "weight");
        size_t weightSize = param.iWidth * param.iHeight * param.iDepth * param.oWidth * param.oHeight * param.oDepth;
        for (int i = 0; i < weightSize; ++i) {
            float r = static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
            writeXMLTag(xml, "item", r > 0.5f ? 1.0f : -1.0f);
        }
        writeXMLCloseTag(xml, "weight");

        writeXMLOpenTag(xml, "offset");
        writeXMLTag(xml, "item", 0.0f);
        writeXMLCloseTag(xml, "offset");
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

    static void writeXMLInfo(std::ofstream &xml, const LayerParam &param) {
        writeXMLTag(xml, "kernelName", param.kernelName);
        writeXMLOpenTag(xml, "workGroupSize");
        for (size_t i = 0; i < sizeof(param.workGroupSize) / sizeof(size_t); ++i) {
            writeXMLTag(xml, "item", param.workGroupSize[i]);
        }
        writeXMLCloseTag(xml, "workGroupSize");
        writeXMLTag(xml, "iWidth", param.iWidth);
        writeXMLTag(xml, "iHeight", param.iHeight);
        writeXMLTag(xml, "iDepth", param.iDepth);
        writeXMLTag(xml, "kernelSize", param.kernelSize);
        writeXMLTag(xml, "oWidth", param.oWidth);
        writeXMLTag(xml, "oHeight", param.oHeight);
        writeXMLTag(xml, "oDepth", param.oDepth);
        writeXMLTag(xml, "oWidthTile", param.oWidthTile);
        writeXMLTag(xml, "oHeightTile", param.oHeightTile);
        writeXMLTag(xml, "oDepthTile", param.oDepthTile);
        writeXMLTag(xml, "iDepthTile", param.iDepthTile);
    }

    /***********************************************************************
     Helper function to generate OpenCL kernel file.
    ************************************************************************/
    static void writeDefine(FILE *o, const std::string &macro, size_t value) {
        fprintf(o, "#define %s %u\n", macro.c_str(), (unsigned int)value);
    }

    static void writeDefine(FILE *o, const std::string &macro, const std::string &value) {
        fprintf(o, "#define %s %s\n", macro.c_str(), value.c_str());
    }

    static void writeUndef(FILE *o, const std::string &macro) {
        fprintf(o, "#undef %s\n", macro.c_str());
    }

    static std::string fileToString(const std::string &fn) {
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

    static void writeKernelDefine(FILE *kernel, const LayerParam &param, size_t idx, Flag flag) {
        writeDefine(kernel, "KERNEL_SIZE", param.kernelSize);
        writeDefine(kernel, "KERNEL_LEN", param.kernelSize * param.kernelSize);
        writeDefine(kernel, "IWIDTH", param.iWidth);
        writeDefine(kernel, "IHEIGHT", param.iHeight);
        writeDefine(kernel, "IDEPTH", param.iDepth);
        writeDefine(kernel, "IN_SIZE", param.iWidth * param.iHeight * param.iDepth);
        writeDefine(kernel, "OWIDTH", param.oWidth);
        writeDefine(kernel, "OHEIGHT", param.oHeight);
        writeDefine(kernel, "ODEPTH", param.oDepth);
        writeDefine(kernel, "OWIDTH_TILE", param.oWidthTile);
        writeDefine(kernel, "OHEIGHT_TILE", param.oHeightTile);
        writeDefine(kernel, "ODEPTH_TILE", param.oDepthTile);
        writeDefine(kernel, "IDEPTH_TILE", param.iDepthTile);
        writeDefine(kernel, "OUT_SIZE", param.oWidth * param.oHeight * param.oDepth);
        writeDefine(kernel, "WORK_GROUP_DIM_0", param.workGroupSize[0]);
        writeDefine(kernel, "WORK_GROUP_DIM_1", param.workGroupSize[1]);
        writeDefine(kernel, "WORK_GROUP_DIM_2", param.workGroupSize[2]);
        writeDefine(kernel, "KERNEL_NAME", param.kernelName);

        std::stringstream ss;
        if (!(flag & FRONT)) {
            fprintf(kernel, "#define in buf%zu\n", idx);
        }
        else {
            ss << "__global float *in, ";
        }

        if (!(flag & BACK)) {
            fprintf(kernel, "#define out buf%zu\n", idx + 1);
        }
        else {
            ss << "__global float *out,";
        }
        writeDefine(kernel, "KERNEL_PARAM", ss.str());
    }

    static void writeKernelUndefine(FILE *kernel, Flag flag) {
        if (!(flag &FRONT)) {
            writeUndef(kernel, "in");
        }
        if (!(flag & BACK)) {
            writeUndef(kernel, "out");
        }
        writeUndef(kernel, "KERNEL_SIZE");
        writeUndef(kernel, "KERNEL_LEN");
        writeUndef(kernel, "IWIDTH");
        writeUndef(kernel, "IHEIGHT");
        writeUndef(kernel, "IDEPTH");
        writeUndef(kernel, "IN_SIZE");
        writeUndef(kernel, "OWIDTH");
        writeUndef(kernel, "OHEIGHT");
        writeUndef(kernel, "ODEPTH");
        writeUndef(kernel, "OWIDTH_TILE");
        writeUndef(kernel, "OHEIGHT_TILE");
        writeUndef(kernel, "ODEPTH_TILE");
        writeUndef(kernel, "IDEPTH_TILE");
        writeUndef(kernel, "OUT_SIZE");
        writeUndef(kernel, "WORK_GROUP_DIM_0");
        writeUndef(kernel, "WORK_GROUP_DIM_1");
        writeUndef(kernel, "WORK_GROUP_DIM_2");
        writeUndef(kernel, "KERNEL_NAME");
        writeUndef(kernel, "KERNEL_PARAM");
    }

    /********************************************************************************************
     Some constant value.
     ********************************************************************************************/
    static const std::string activateFunc;
    static const std::string convKernel;
    static const std::string poolKernel;
    static const std::string fullKernel;
    static const std::string rbfKernel;
};

/* Initialize the constant value. */
const std::string CNNGenerator::activateFunc = "\
float sigmod(float in) {\n\
    return 1.0f / (1.0f + exp(-in)); \n\
}";

const std::string CNNGenerator::convKernel = CNNGenerator::fileToString("convolution.cl");
const std::string CNNGenerator::poolKernel = CNNGenerator::fileToString("pool.cl");
const std::string CNNGenerator::fullKernel = CNNGenerator::fileToString("full.cl");
const std::string CNNGenerator::rbfKernel = CNNGenerator::fileToString("rbf.cl");

int main(int argc, char *argv[]) {

    CNNGenerator::LayerParam paramsUntile[] = {
        {
            CNNGenerator::CONV,
            "conv1",
            {28, 28, 3},
            32,
            32,
            1,
            5,
            28,
            28,
            6,
            1,
            1,
            1,
            1
        },
        {
            CNNGenerator::POOL,
            "pool2",
            { 14, 14, 2 },
            28,
            28,
            6,
            2,
            14,
            14,
            6,
            1,
            1,
            1,
            1
        },
        {
            CNNGenerator::CONV,
            "conv3",
            { 16, 1, 1 },
            14,
            14,
            6,
            5,
            10,
            10,
            16,
            1,
            1,
            1,
            1
        },
        {
            CNNGenerator::POOL,
            "pool4",
            { 16, 1, 1 },
            10,
            10,
            16,
            2,
            5,
            5,
            16,
            1,
            1,
            1,
            1
        },
        {
            CNNGenerator::CONV,
            "conv5",
            { 16, 1, 1 },
            5,
            5,
            16,
            5,
            1,
            1,
            120,
            1,
            1,
            1,
            1
        },
        {
            CNNGenerator::FULL,
            "full6",
            { 12, 1, 1 },
            1,
            1,
            120,
            10,
            84,
            1,
            1,
            1,
            1,
            1,
            1
        },
        {
            CNNGenerator::RBF,
            "rbf7",
            { 10, 1, 1 },
            84,
            1,
            1,
            14,
            10,
            1,
            1,
            1,
            1,
            1,
            1
        }
    };

    CNNGenerator::LayerParam paramsTile[] = {
        {
            CNNGenerator::CONV,
            "conv1",
            {28, 28, 3},
            32,
            32,
            1,
            5,
            28,
            28,
            6,
            1,
            1,
            1,
            1
        },
        {
            CNNGenerator::POOL,
            "pool2",
            { 14, 14, 2 },
            28,
            28,
            6,
            2,
            14,
            14,
            6,
            1,
            1,
            1,
            1
        },
        {
            CNNGenerator::CONV,
            "conv3",
            { 16, 1, 1 },
            14,
            14,
            6,
            5,
            10,
            10,
            16,
            1,
            1,
            1,
            1
        },
        {
            CNNGenerator::POOL,
            "pool4",
            { 16, 1, 1 },
            10,
            10,
            16,
            2,
            5,
            5,
            16,
            1,
            1,
            1,
            1
        },
        {
            CNNGenerator::CONV,
            "conv5",
            { 16, 1, 1 },
            5,
            5,
            16,
            5,
            1,
            1,
            120,
            1,
            1,
            1,
            1
        },
        {
            CNNGenerator::FULL,
            "full6",
            { 12, 1, 1 },
            1,
            1,
            120,
            10,
            84,
            1,
            1,
            1,
            1,
            1,
            1
        },
        {
            CNNGenerator::RBF,
            "rbf7",
            { 10, 1, 1 },
            84,
            1,
            1,
            14,
            10,
            1,
            1,
            1,
            1,
            1,
            1
        }
    };

    /*CNNGenerator::genCNN("../cnn/kernel/conv1.xml", "../cnn/kernel/conv1.cl", 1, &paramsUntile[0]);
    CNNGenerator::genCNN("../cnn/kernel/pool2.xml", "../cnn/kernel/pool2.cl", 1, &paramsUntile[1]);
    CNNGenerator::genCNN("../cnn/kernel/full6.xml", "../cnn/kernel/full6.cl", 1, &paramsUntile[5]);
    CNNGenerator::genCNN("../cnn/kernel/rbf7.xml", "../cnn/kernel/rbf7.cl", 1, &paramsUntile[6]);
    CNNGenerator::genCNN("../cnn/kernel/lenet5.xml", "../cnn/kernel/lenet5.cl", 7, paramsUntile);*/

    CNNGenerator::genCNN("../cnn/kernel/conv1_tile.xml", "../cnn/kernel/conv1_tile.cl", 1, &paramsTile[0]);

    return 0;
}