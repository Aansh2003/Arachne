#ifndef Relu_H
#define Relu_H

#include "Activation.hpp"
#include "Tensor.hpp"

class Relu : public Activation
{
    public:
        Relu(std::pair<int,int> inputSize, std::string type = "relu");
        Tensor<float> forward(Tensor<float>) override;
        void backward() override;

        std::string type;
    private:
        static void relu(float *);
        static void leakyrelu(float *);
};

Relu::Relu(std::pair<int,int> inputSize,std::string type) : Activation(inputSize,type), type(type)
{

}

Tensor<float> Relu::forward(Tensor<float> input)
{
    if(type=="relu")
    {
        input.map(Relu::relu);
    }
    else if(type=="leaky")
    {
        input.map(Relu::leakyrelu);
    }
    else
    {
        throw std::runtime_error("Invalid ReLu type");
    }
    return input;
}

void Relu::backward()
{

}

void Relu::relu(float* input)
{
    if(*input<0)
        *input = 0;
}

void Relu::leakyrelu(float* input)
{
    if(*input<0)
        *input = -0.1 * *input;
}

#endif