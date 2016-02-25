#include "convolution.hpp"

using namespace cnn;

#define NUM_TEST 10

namespace test {

    // Run the test.
    void runTimeTest(const std::string &xmlFile,
        const std::string &clBinaryFile,
        const std::string &outFile,
        DeviceType type
        ) {

        std::ofstream o(outFile.c_str());
        if (!o.is_open()) {
            std::cerr << "Can't open file " << outFile << std::endl;
            exit(-1);
        }
        o << "<?xml version=\"1.0\" encoding=\"utf-8\"?>" << std::endl;
        writeXMLOpenTag(o, "results");
        
        // Parse the convolution xml file.
        std::string str = fileToString(xmlFile);
        char *text = new char[str.size() + 1];
        memcpy((void *)text, (void *)(&str[0]), str.size() * sizeof(char));
        text[str.size()] = '\0';
        rapidxml::xml_document<> doc;
        doc.parse<0>(text);
        rapidxml::xml_node<> *root = doc.first_node();

        // For each convolutional layer.
        for (rapidxml::xml_node<> *node = root->first_node(); node; node = node->next_sibling()) {

            cnn::ConvolutionLayer layer = cnn::createConvolutionLayerFromXML(node, type, clBinaryFile);
            cnn::vec input(layer.iWidth * layer.iHeight * layer.iDepth);
            for (int i = 0; i < input.size(); ++i) {
                input[i] = (float)i;
            }

            unsigned long long totalTime = 0;
            for (int i = 0; i < NUM_TEST; ++i) {
                totalTime += layer.forward(input);
            }
            std::cout << "Average Time: " << totalTime / NUM_TEST << std::endl;
            
            writeXMLOpenTag(o, "result");
            writeXMLTag(o, "iWidth", layer.iWidth);
            writeXMLTag(o, "iHeight", layer.iHeight);
            writeXMLTag(o, "iDepth", layer.iDepth);
            writeXMLTag(o, "kernelSize", layer.kernelSize);
            writeXMLTag(o, "oDepth", layer.oDepth);
            writeXMLTag(o, "averageTime", (float)totalTime / (float)NUM_TEST);
            writeXMLCloseTag(o, "result");

        }
        writeXMLCloseTag(o, "results");
        std::cout << "Finish testing!" << std::endl;

        delete[] text;
        o.close();
    }

    void runFuncTest(const std::string &xmlFile,
        const std::string &clBinaryFile,
        const std::string &outFile,
        DeviceType type
        ) {

        std::ofstream o(outFile.c_str());
        if (!o.is_open()) {
            std::cerr << "Can't open file " << outFile << std::endl;
            exit(-1);
        }
        o << "<?xml version=\"1.0\" encoding=\"utf-8\"?>" << std::endl;
        writeXMLOpenTag(o, "results");

        // Parse the convolution xml file.
        std::string str = fileToString(xmlFile);
        char *text = new char[str.size() + 1];
        memcpy((void *)text, (void *)(&str[0]), str.size() * sizeof(char));
        text[str.size()] = '\0';
        rapidxml::xml_document<> doc;
        doc.parse<0>(text);
        rapidxml::xml_node<> *root = doc.first_node();

        // For each convolutional layer.
        for (rapidxml::xml_node<> *node = root->first_node(); node; node = node->next_sibling()) {

            cnn::ConvolutionLayer layer = cnn::createConvolutionLayerFromXML(node, type, clBinaryFile);
            cnn::vec input(layer.iWidth * layer.iHeight * layer.iDepth);
            for (int i = 0; i < input.size(); ++i) {
                input[i] = (float)rand() / (float)RAND_MAX - 0.5f;
            }

            layer.forward(input);
            cnn::vec output(layer.output);
            layer.forwardCPU(input);
            for (int i = 0; i < layer.output.size(); ++i) {
                assert(abs(output[i] - layer.output[i]) < 0.0001f);
            }
            //dumpVec(o, layer.output, layer.oWidth, layer.oHeight, layer.oDepth);
        }
        writeXMLCloseTag(o, "results");
        std::cout << "Finish testing!" << std::endl;

        delete[] text;
        o.close();
    }
}