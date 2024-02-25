#ifndef Activation_H
#define Activation_H

#include <functional>
#include "Tensor.hpp"
#include <unordered_map>

class Activation
{
    public:
        static void relu(float*);
        static void Relu(Tensor<float>&);
        static void Linear(Tensor<float>&);
    private:
};

void Activation::Relu(Tensor<float>& input)
{
    input.map(Activation::relu);
}

void Activation::relu(float* input)
{
    if(*input<0)
        *input = 0;
}

void Activation::Linear(Tensor<float>& input)
{
    
}

#endif