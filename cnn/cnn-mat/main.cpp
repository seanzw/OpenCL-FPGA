#include "convolution.hpp"
#include <iostream>

#define TEST 1000

int main() {
    cnn::ConvolutionLayer layer(32, 32, 1, 5, 6);

    unsigned long long time = 0;
    for (int i = 0; i < TEST; ++i) {
        time += layer.forwardCL();
    }

    std::cout << "Average time: " << time / TEST << std::endl;

    return 0;
}
