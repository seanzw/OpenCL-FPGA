#include "convolution.hpp"
#include <iostream>

int main(int argc, char *argv[]) {

    cnn::ConvolutionLayer layer = cnn::createConvolutionLayerFromXML("convolutional1.xml");
    cnn::vec input(layer.iWidth * layer.iHeight * layer.iDepth);
    for (int i = 0; i < input.size(); ++i) {
        input[i] = (float)i;
    }

    //layer.forward(input);

    layer.forwardGPU(input);

    std::cout << "Finish testing!" << std::endl;

    return 0;
}
