#ifndef Optimizer_H
#define Optimizer_H

#include "Tensor.hpp"

class Optimizer
{
    public:
    virtual void update_weights(Tensor<float>&,float) = 0;
};

#endif