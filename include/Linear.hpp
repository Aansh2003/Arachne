// Linear.hpp
#ifndef LINEAR_H
#define LINEAR_H

#include "Model.hpp"
#include <utility>
#include "Activation.hpp"

class Linear : public Model {
public:
    Linear(std::pair<int,int> inputSize, std::pair<int,int> outputSize, std::string="relu");
    int getParamCount() override;
    std::pair<int,int> getInputSize() override;
    std::pair<int,int> getOutputSize() override;
    Tensor<float> forward(Tensor<float>) override;

    void backward() override;
private:
    std::pair<int,int> inputSize;
    std::pair<int,int> outputSize;
    int paramCount;
};

Linear::Linear(std::pair<int,int> inputSize, std::pair<int,int> outputSize, std::string activation) : Model("Linear",activation),inputSize(inputSize), outputSize(outputSize)
{
    paramCount = inputSize.second * outputSize.first;
}

int Linear::getParamCount()
{
    return paramCount;
}

Tensor<float> Linear::forward(Tensor<float> input)
{

}

void Linear::backward()
{
}

std::pair<int,int> Linear::getOutputSize()
{
    return outputSize;
}

std::pair<int,int> Linear::getInputSize()
{
    return inputSize;
}

#endif
