#ifndef Loss_H
#define Loss_H

#include "Tensor.hpp"
#include <cmath>

class Loss
{
    public:
        static float MSELoss(Tensor<float>,Tensor<float>);
        static float MAELoss(Tensor<float>,Tensor<float>);
};

float Loss::MSELoss(Tensor<float>prediction,Tensor<float>actual)
{
    if(prediction.getSize() != actual.getSize())
        throw std::runtime_error("Invalid dimensions");

    float sum = 0;

    for(int i=0;i<prediction.getSize().first;i++)
    {
        for(int j=0;j<prediction.getSize().second;j++)
        {
            sum += pow(prediction.data[i][j]-actual.data[i][j],2);
        }
    }

    return sum/(prediction.getSize().first * prediction.getSize().second);
}

float Loss::MAELoss(Tensor<float>prediction,Tensor<float>actual)
{
    if(prediction.getSize() != actual.getSize())
        throw std::runtime_error("Invalid dimensions");

    float sum = 0;

    for(int i=0;i<prediction.getSize().first;i++)
    {
        for(int j=0;j<prediction.getSize().second;j++)
        {
            sum += abs(prediction.data[i][j]-actual.data[i][j]);
        }
    }

    return sum/(prediction.getSize().first * prediction.getSize().second);
}

#endif