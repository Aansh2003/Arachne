// Linear.hpp
#ifndef NORMALIZE_H
#define NORMALIZE_H

#include "Model.hpp"
#include <utility>
#include "Activation.hpp"
#include <cmath>

class Normalize : public Model {
public:
    Normalize(std::pair<int,int> inputSize);
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

Normalize::Normalize(std::pair<int,int> inputSize) : Model("Normalization"), inputSize(inputSize)
{
    paramCount = 0;
    this->inputSize = make_pair(inputSize.first,inputSize.second);
    this->outputSize = make_pair(inputSize.first,inputSize.second);
}

int Normalize::getParamCount()
{
    return paramCount;
}

Tensor<float> Normalize::forward(Tensor<float> input)
{
    float max = input.max();
    float min = input.min();
    Tensor<float> newCopy = input.copy();
    for(int i=0;i<newCopy.getSize().first;i++)
    {
        for(int j=0;j<newCopy.getSize().second;j++)
        {
             newCopy.data[i][j] = (newCopy.data[i][j] - min)/(max-min);
        }
    }
    
    return newCopy;
}

void Normalize::backward()
{
}

std::pair<int,int> Normalize::getOutputSize()
{
    return outputSize;
}

std::pair<int,int> Normalize::getInputSize()
{
    return inputSize;
}


#endif
