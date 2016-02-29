#include "convolution.hpp"
#include "test.hpp"
#include <iostream>

int main(int argc, char *argv[]) {

    if (argc != 3 && argc != 4) {
        std::cout << "Usage: cnn <xml> <result> [xclbin]" << std::endl;
        exit(-1);
    }

    CNN *cnn;

    std::string xmlFile(argv[1]);
    std::string testFile(argv[2]);

    if (argc == 4) {
        std::string xclbinFile(argv[3]);
        cnn = new CNN(xmlFile, xclbinFile);
    }
    else {
        cnn = new CNN(xmlFile);
    }

    cnn::vec in(cnn->getInSize());
    for (int i = 0; i < in.size(); ++i) {
        in[i] = (float)rand() / (float)RAND_MAX - 0.5f;
    }

    test::runFuncTest(cnn, in);
    test::runTimeTest(cnn, in, testFile);
    
    return 0;
}
