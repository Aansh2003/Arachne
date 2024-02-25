#ifndef Activation_H
#define Activation_H

#include <functional>
#include "Tensor.hpp"
#include <unordered_map>

class Activation
{
    public:
        static void relu(float*);
    private:
};

void Activation::relu(float* input)
{
    if(*input<0)
        *input = 0;
}

#endif