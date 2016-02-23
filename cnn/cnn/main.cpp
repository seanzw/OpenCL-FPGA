#include "convolution.hpp"
#include <iostream>

int main(int argc, char *argv[]) {

    if (argc != 3) {
        std::cout << "Usage: cnn <type> <clFileName>" << std::endl;
        exit(-1);
    }

    cnn::DeviceType type;
    std::string clFileName(argv[2]);
    if (argv[1] == "fpga") {
        type = cnn::FPGA;
    }
    else if (argv[1] == "gpu") {
        type = cnn::GPU;
    }
    else {
        type = cnn::CPU;
    }

    cnn::ConvolutionLayer layer = cnn::createConvolutionLayerFromXML("convolutional1.xml", type, clFileName);
    cnn::vec input(layer.iWidth * layer.iHeight * layer.iDepth);
    for (int i = 0; i < input.size(); ++i) {
        input[i] = (float)i;
    }

    layer.forward(input);

    std::cout << "Finish testing!" << std::endl;

    return 0;
}
