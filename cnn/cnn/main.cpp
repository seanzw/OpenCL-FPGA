#include "convolution.hpp"
#include "test.hpp"
#include <iostream>

int main(int argc, char *argv[]) {

    if (argc != 5) {
        std::cout << "Usage: cnn <type> <clBinaryFile> <xmlList> <testFile>" << std::endl;
        exit(-1);
    }

    cnn::DeviceType type;
    std::string deviceType(argv[1]);
    std::string clFile(argv[2]);
    std::string xmlFile(argv[3]);
    std::string testFile(argv[4]);
    if (deviceType == "fpga") {
        type = cnn::FPGA;
    }
    else if (deviceType == "gpu") {
        type = cnn::GPU;
    }
    else {
        type = cnn::CPU;
    }

    test::runFuncTest(xmlFile, clFile, "output.xml", type);
    test::runTimeTest(xmlFile, clFile, testFile, type);

    return 0;
}
