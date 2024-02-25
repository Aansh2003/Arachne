#ifndef Model_H
#define Model_H

#include "Tensor.hpp" 
#include <string>
#include "Activation.hpp"

class Model {
public:
    Model(const std::string& _type = "model", const std::string& _activation = "relu") : type(_type),activation(_activation) {}
    virtual Tensor<float> forward(Tensor<float>) = 0;
    virtual void backward() = 0;
    virtual ~Model() {}
    virtual int getParamCount() = 0;
    virtual std::pair<int,int> getInputSize() = 0;
    virtual std::pair<int,int> getOutputSize() = 0;

    std::string activation;
    std::string type;
};

#endif
