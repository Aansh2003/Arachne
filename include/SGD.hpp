#ifndef SGD_H
#define SGD_H

#include "Optimizer.hpp"

class SGD : public Optimizer
{
    public:
    SGD(float);
    void update_weights(Tensor<float>&,Tensor<float>,Tensor<float>) override;

    private:
    float learning_rate;
};

SGD::SGD(float alpha=1e-2) : learning_rate(alpha) {}

void SGD::update_weights(Tensor<float>& weights, Tensor<float> gradient, Tensor<float> output)
{
    weights = weights - output * gradient;
}

#endif