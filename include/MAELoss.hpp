#ifndef Mae_loss_H
#define Mae_loss_H

#include "Loss.hpp"

class MAELoss : public Loss
{
    public:
    float loss(Tensor<float>,Tensor<float>) override;
    Tensor<float> derivative(Tensor<float>,Tensor<float>) override;
};

float MAELoss::loss(Tensor<float> prediction,Tensor<float> actual)
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

Tensor<float> MAELoss::derivative(Tensor<float> prediction,Tensor<float> actual)
{
    
}

#endif