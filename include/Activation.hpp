#ifndef Activation_H
#define Activation_H

#include <functional>
#include "Tensor.hpp"
#include <unordered_map>
#include "Variables.hpp"
#include <cmath>

class Activation
{
    public:
        static void relu(float*);
        static void Relu(Tensor<float>&);
        static void Linear(Tensor<float>&);
        static void Softmax(Tensor<float>&);
        static void Softmax2d(Tensor<float>&); // Not implemented
    private:
};

// Calls a map to the relu helper function
void Activation::Relu(Tensor<float>& input)
{
    input.map(Activation::relu);
}

// InPlace edits the values using f(x) = { x if x>0
//                                       { 0 else
void Activation::relu(float* input)
{
    if(*input<0)
        *input = 0;
}

// No influence on tensor f(x) = x
void Activation::Linear(Tensor<float>& input)
{
    
}

void Activation::Softmax(Tensor<float>& input)
{
    for(int i=0; i<input.getSize().first;i++)
    {
    	float sum = 0;
    	for(int j=0; j<input.getSize().second; j++)
    	{
	    sum += pow(Variables::e,input.data[i][j]);
    	}
    	for(int j=0;j<input.getSize().second;j++)
    	{
    	    input.data[i][j] = pow(Variables::e,input.data[i][j]) / sum;
    	}
    }
}

void Activation::Softmax2d(Tensor<float>& input)
{
    float sum = 0;
    for(int i=0; i<input.getSize().first;i++)
    {
    	for(int j=0; j<input.getSize().second; j++)
    	{
	    sum += pow(Variables::e,input.data[i][j]);
    	}
    }
    for(int i=0; i<input.getSize().first;i++)
    {
        for(int j=0;j<input.getSize().second;j++)
    	{
    	    input.data[i][j] = pow(Variables::e,input.data[i][j]) / sum;
    	}
    }

}
#endif
