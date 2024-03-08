#ifndef Model_H
#define Model_H

#include "Tensor.hpp" 
#include <string>

class Model {
public:
    Model(const std::string& _type = "model", const bool& trainable = false,Tensor<float>* weights = NULL) : type(_type),trainable(trainable),weights(weights) {}
    virtual Tensor<float> forward(Tensor<float>) = 0;
    virtual void backward() = 0;
    virtual ~Model() {}
    virtual int getParamCount() = 0;
    virtual std::pair<int,int> getInputSize() = 0;
    virtual std::pair<int,int> getOutputSize() = 0;
    bool trainable;
    std::string type;
    Tensor<float>* weights;
};

#endif
