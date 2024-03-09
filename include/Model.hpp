#ifndef Model_H
#define Model_H

#include "Tensor.hpp" 
#include <string>

class Model {
public:
    Model(const std::string& _type = "model", const bool& trainable = false,Tensor<float>* weights = NULL) : type(_type),trainable(trainable),weights(weights) {}
    virtual Tensor<float> forward(Tensor<float>) = 0;
    void backward(Tensor<float>);
    void computeGradients(Tensor<float>);
    virtual ~Model() {}
    virtual int getParamCount() = 0;
    virtual std::pair<int,int> getInputSize() = 0;
    virtual std::pair<int,int> getOutputSize() = 0;
    bool trainable;
    std::string type;
    bool isforward = false;
    Tensor<float>* weights;
    Tensor<float>* gradients;
    Tensor<float>* inputs;
};

void Model::backward(Tensor<float> gradient)
{
    if(!isforward)
        throw std::runtime_error("Forward pass must be called before backward pass");
    computeGradients(gradient);
}

void Model::computeGradients(Tensor<float> gradient)
{
    gradient.transpose();
    gradients = new Tensor(*inputs * gradient);\
}

#endif
