#ifndef LAYER_HEADER
#define LAYER_HEADER

#include "util.hpp"
#include <cmath>

namespace cnn {

    enum DeviceType {
        CPU,
        GPU,
        FPGA
    };

    class Layer {
    public:

        Layer(size_t iWidth,
            size_t iHeight,
            size_t iDepth,
            size_t oWidth,
            size_t oHeight,
            size_t oDepth,
            const vec &weight,
            const vec &offset,
            DeviceType type
            ) : iWidth(iWidth),
            iHeight(iHeight),
            iDepth(iDepth),
            oWidth(oWidth),
            oHeight(oHeight),
            oDepth(oDepth),
            weight(weight),
            offset(offset),
            type(type) {}

        virtual unsigned long long forward(const vec &in) = 0;

        size_t iWidth;
        size_t iHeight;
        size_t iDepth;

        size_t oWidth;
        size_t oHeight;
        size_t oDepth;

        // Sigmod function.
        float sigmod(float i) {
            return 1.0f / (1.0f + expf(-i));
        }

        // Weight and offset.
        vec weight;
        vec offset;

        // Output vector.
        vec output;

        // Pointer to next layer.
        Layer *next;

        // Device type.
        const DeviceType type;
    };
}

#endif