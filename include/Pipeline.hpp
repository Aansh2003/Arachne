#ifndef Pipeline_H
#define Pipeline_H

#include "Model.hpp"
#include <vector>
#include <iostream>
#include <utility>
#include <unordered_map>

class Pipeline {
public:
    Pipeline();
    void add(Model*);
    void printPipeline();

private:
    std::vector<Model*> network;
};

Pipeline::Pipeline()
{

}

void Pipeline::add(Model* model)
{
    if(network.size()>0)
    {
        Model* current = network.back();
        std::pair<int,int> current_size = current->getOutputSize();
        if (current_size != model->getInputSize())
            throw std::runtime_error("Incorrect size of weights, input size must match previous output size.");
    }
    network.push_back(model);
}

void Pipeline::printPipeline()
{
    int total_parameter_count = 0;
    std::unordered_map<std::string,int> count_check;
    std::cout<<"Layer"<<"\t\t"<<"Input"<<"\t\t"<<"Output"<<"\t\t"<<"Activation"<<"\t\t"<<"Parameter Count"<<std::endl;
    for(Model* model: network)
    {
        total_parameter_count += model->getParamCount();
        count_check[model->type]+=1;
        std::cout<<model->type<<" "<<count_check[model->type]<<":"<<"\t\t"<<"("<<model->getInputSize().first<<","<<model->getInputSize().second<<")"<<"\t\t"<<"("<<model->getOutputSize().first<<","<<model->getOutputSize().second<<")"<<"\t\t"<<model->activation<<"\t\t"<<model->getParamCount()<<std::endl;
    }
    std::cout<<"Total Parameter Count:"<<"\t"<<total_parameter_count<<std::endl;
}

#endif