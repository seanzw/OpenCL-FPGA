#include "CNNGenerator.hpp"

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