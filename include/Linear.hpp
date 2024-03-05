// Linear.hpp
#ifndef LINEAR_H
#define LINEAR_H

#include "Model.hpp"
#include <utility>
#include "Activation.hpp"
#include <cmath>

class Linear : public Model {
public:
    Linear(std::pair<int,int> inputSize, int outputSize, void (*activation_function)(Tensor<float>&));
    int getParamCount() override;
    std::pair<int,int> getInputSize() override;
    std::pair<int,int> getOutputSize() override;
    Tensor<float> forward(Tensor<float>) override;
    void printWeights();
    void backward() override;
    Tensor<float>* weights;

private:
    std::pair<int,int> inputSize;
    std::pair<int,int> outputSize;
    int paramCount;
    std::pair<int,int> weight_size;
    void (*act_func)(Tensor<float>&);
};

Linear::Linear(std::pair<int,int> inputSize, int outputSize, void (*activation_function)(Tensor<float>&)) : Model("Linear",true),inputSize(inputSize), outputSize(make_pair(inputSize.second,outputSize)),act_func(activation_function)
{
    weight_size = make_pair(outputSize,inputSize.second);
    paramCount = weight_size.second * weight_size.first;
    float value = std::sqrt(6.0/((inputSize.first*inputSize.second)+(outputSize*inputSize.second)));
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
    Tensor<float> next_val = weights->multiply(input);
    act_func(next_val);
    return next_val;
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
