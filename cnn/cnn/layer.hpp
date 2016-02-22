#ifndef LAYER_HEADER
#define LAYER_HEADER

#include "util.hpp"

namespace cnn {
    class Layer {
    public:

        Layer(size_t iWidth, size_t iHeight, size_t iDepth,
            size_t oWidth, size_t oHeight, size_t oDepth,
            const vec &weight, const vec &offset
            ) : iWidth(iWidth), iHeight(iHeight), iDepth(iDepth),
            oWidth(oWidth), oHeight(oHeight), oDepth(oDepth),
            weight(weight), offset(offset) {}

        virtual void forward(const vec &in) = 0;

        size_t iWidth;
        size_t iHeight;
        size_t iDepth;

        size_t oWidth;
        size_t oHeight;
        size_t oDepth;

        // Sigmod function.
        float sigmod(float i) {
            return 1.0f / (1.0f + std::expf(-i));
        }

        // Weight and offset.
        vec weight;
        vec offset;

        // Output vector.
        vec output;

        // Pointer to next layer.
        Layer *next;


    };
}

#endif