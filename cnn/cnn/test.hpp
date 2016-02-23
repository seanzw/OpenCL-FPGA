#include "convolution.hpp"

using namespace cnn;

#define NUM_TEST 1000

namespace test {

    // Run the test.
    void runTest(const std::string &xmlFile,
        const std::string &clFile,
        const std::string &outFile,
        DeviceType type
        ) {
        
        cnn::ConvolutionLayer layer = cnn::createConvolutionLayerFromXML(xmlFile, type, clFile);
        cnn::vec input(layer.iWidth * layer.iHeight * layer.iDepth);
        for (int i = 0; i < input.size(); ++i) {
            input[i] = (float)i;
        }

        unsigned long long totalTime = 0;
        for (int i = 0; i < NUM_TEST; ++i) {
            totalTime += layer.forward(input);
        }
        std::cout << "Average Time: " << totalTime / NUM_TEST << std::endl;
        std::cout << "Finish testing!" << std::endl;

        std::ofstream o(outFile);
        if (!o.is_open()) {
            std::cerr << "Can't open file " << outFile << std::endl;
            exit(-1);
        }

        o << "<?xml version=\"1.0\" encoding=\"utf-8\"?>" << std::endl;
        writeXMLOpenTag(o, "result");
        writeXMLTag(o, "iWidth", layer.iWidth);
        writeXMLTag(o, "iHeight", layer.iHeight);
        writeXMLTag(o, "iDepth", layer.iDepth);
        writeXMLTag(o, "kernelSize", layer.kernelSize);
        writeXMLTag(o, "oDepth", layer.oDepth);
        writeXMLTag(o, "averageTime", totalTime / NUM_TEST);
        writeXMLCloseTag(o, "result");

        o.close();
    }
}