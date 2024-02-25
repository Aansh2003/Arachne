// Linear.hpp
#ifndef LINEAR_H
#define LINEAR_H

#include "Model.hpp"
#include <utility>
#include "Activation.hpp"
#include <cmath>

class Linear : public Model {
public:
    Linear(std::pair<int,int> inputSize, std::pair<int,int> outputSize, std::string="relu");
    int getParamCount() override;
    std::pair<int,int> getInputSize() override;
    std::pair<int,int> getOutputSize() override;
    Tensor<float> forward(Tensor<float>) override;
    void printWeights();
    void backward() override;
private:
    std::pair<int,int> inputSize;
    std::pair<int,int> outputSize;
    int paramCount;
    Tensor<float>* weights;
    std::pair<int,int> weight_size;
};

Linear::Linear(std::pair<int,int> inputSize, std::pair<int,int> outputSize, std::string activation) : Model("Linear",activation),inputSize(inputSize), outputSize(outputSize)
{
    paramCount = inputSize.second * outputSize.first;
    weight_size = make_pair(inputSize.second,outputSize.first);
    float value = std::sqrt(6.0/((inputSize.first*inputSize.second)+(outputSize.first*outputSize.second)));
    float **data = new float*[weight_size.first];
    for(int i=0;i<weight_size.first;i++)
    {
        data[i] = new float[weight_size.second];
    }

    for(int i=0;i<weight_size.first;i++)
    {
        for(int j=0;j<weight_size.second;j++)
        {
            data[i][j] = value;
        }
    }

    weights = new Tensor<float>(data,weight_size);
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

void Linear::printWeights()
{
    for(int i=0;i<weight_size.first;i++)
    {
        for(int j=0;j<weight_size.second;j++)
        {
            std::cout<<this->weights->data[i][j];
            if(j!=weight_size.second-1)
                std::cout<<",";
        }
        std::cout<<std::endl;
    }
}

#endif
