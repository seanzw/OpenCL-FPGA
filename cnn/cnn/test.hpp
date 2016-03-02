#include "cnn.hpp"

using namespace cnn;

#define NUM_TEST 10

namespace test {

    // Run the test.
    void runTimeTest(CNN *cnn, const vec &in, const std::string &outFile
        ) {

        std::ofstream o(outFile.c_str());
        if (!o.is_open()) {
            std::cerr << "Can't open file " << outFile << std::endl;
            exit(-1);
        }
        o << "<?xml version=\"1.0\" encoding=\"utf-8\"?>" << std::endl;
        writeXMLOpenTag(o, "results");
        
        
        
        writeXMLOpenTag(o, "result");

        unsigned long long totalTime = 0;
        for (size_t i = 0; i < NUM_TEST; ++i) {
            totalTime += cnn->forwardCL(in);
        }
        size_t averageTime = totalTime / NUM_TEST;
        std::cout << "Average Time: " << averageTime << std::endl;
        writeXMLTag(o, "averageTime", averageTime);
        writeXMLCloseTag(o, "result");
        writeXMLCloseTag(o, "results");
        std::cout << "Finish testing!" << std::endl;

        o.close();
    }

    void runFuncTest(CNN *cnn, const vec &in) {

        cnn->forwardCL(in);
        cnn::vec outCL(cnn->getOut());
        cnn->forwardCPU(in);
        cnn::vec outCPU(cnn->getOut());
        for (int i = 0; i < outCL.size(); ++i) {
            if (abs(outCL[i] - outCPU[i]) > 0.0001f) {
                std::cerr << "Failed test: CL kernel works incorrect. " << std::endl;
                exit(-2);
            }
        }

        std::cout << "CL Kernel works perfect!. " << std::endl;
    }
}