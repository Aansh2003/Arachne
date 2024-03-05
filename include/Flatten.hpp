#ifndef Flatten_H
#define Flatten_H

#include "Model.hpp"
#include "Activation.hpp"

class Flatten : public Model {
public:
    Flatten(std::pair<int,int> inputSize);
    int getParamCount() override;
    std::pair<int,int> getInputSize() override;
    std::pair<int,int> getOutputSize() override;
    Tensor<float> forward(Tensor<float>) override;
    void backward() override;
private:
    std::pair<int,int> inputSize;
    std::pair<int,int> outputSize;
    int paramCount;
    void (*act_func)(Tensor<float>&);
};

Flatten::Flatten(std::pair<int,int> inputSize) : Model("Flatten") , inputSize(inputSize) , outputSize(1,inputSize.first*inputSize.second)
{
    act_func = Activation::Linear;
    paramCount = 0;
}

int Flatten::getParamCount()
{
    return paramCount;
}

std::pair<int,int> Flatten::getInputSize()
{
    return inputSize;
}

std::pair<int,int> Flatten::getOutputSize()
{
    return outputSize;
}

Tensor<float> Flatten::forward(Tensor<float> input)
{
    return input.flatten();
}

void Flatten::backward()
{

}

#endif