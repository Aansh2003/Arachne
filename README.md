# Overview

This is a C++ implementation for Deep Learning models. The architecture allows custom model structures and also provides in-built features to aid in easier preprocessing and training. It is currently suited for Datasets however functionality for other data formats(like images and sounds will be added soon.



# Documentation

## Getting started

```
touch src/myfile.cpp
nano src/myfile.cpp
```

Replace nano with any editor of your choice.\

Use `make` at the home directory to start compilation.

## Example usage
### Create a Tensor with random values

Replace the sizes with your required size.
Call the print function to display the contents of the tensor.

```
#include "Tensor.hpp"

int main()
{
    std::pair<int,int> size = make_pair(2,2);
    Tensor<int> myTensor = Tensor<int>::randomTensor(size);
    myTensor.print();
}
```

### Create a Tensor with custom values

```
#include "Tensor.hpp"

int main()
{
    std::pair<int,int> size = make_pair(2,2);

    // Allocate data in a 2d matrix
    float** data = float*[size.first];
    for(int i=0;i<size.first;i++)
    {
        data[i] = new float[size.second];
    }

    // Add required data
    data[0][0] = 0;
    data[0][1] = 1;
    data[1][0] = 2;
    data[1][1] = 3;

    Tensor<float> myTensor = Tensor<float>(data,size);

    // De-allocate created data
    for(int i=0;i<size.first;i++)
    {
        delete[] data[i];
    }
    delete[] data;

    myTensor.print();
}
```

### Create a Tensor from a preprocessed CSV file

CSV file needs to be preprocessed already with required type. Replace **data.csv** with your csv filepath relative to the home directory, or use global pathing.
```
#include "Tensor.hpp"

int main()
{
    Tensor<float> myTensor = Tensor::ReadCSV('data.csv');
    myTensor.print();
}
```

### Split each row of a tensor into individual tensors

```
#include "Tensor.hpp"
#include <vector>

int main()
{
    std::pair<int,int> size = make_pair(2,2);
    Tensor<int> myTensor = Tensor<int>::randomTensor(size);

    vector<Tensor<float>> myTensorList = myTensor.row_split();

    for(auto individual_tensor: myTensorList)
    {
        // Do operations on individual_tensor
    }
}
```


### Separate input and output columns from a given Tensor

```
#include "Tensor.hpp"

int main()
{
    std::pair<int,int> size = make_pair(2,2);
    Tensor<int> myTensor = Tensor<int>::randomTensor(size);

    std::pair<Tensor<float>,Tensor<float>> input_output myTensor.input_output_split();

    // Do operations on input_output.first(input) and input_output.second(output)
}
```

### Creating a simple Pipeline and training a model


```
#include "Tensor.hpp"
#include "Pipeline.hpp"
#include "SGD.hpp"
#include "MSELoss.hpp"
#include "Linear.hpp"
#include "Relu.hpp"

int main()
{
    std::pair<int,int> size = make_pair(4,2);
    Tensor<int> myTensor = Tensor<int>::randomTensor(size);

    Pipeline myPipeline;

    // Add layers and activations to the pipeline

    Linear* q = new Linear(make_pair(4,2),3);
    Relu* r = new Relu(make_pair(4,3));
    Linear* d = new Linear(make_pair(4,3),2);
    Relu *e = new Relu(make_pair(4,2));

    myPipeline.add(q);
    myPipeline.add(r);
    myPipeline.add(d);
    myPipeline.add(e);

    myPipeline.printPipeline();

    // Define optimizer and Loss function

    SGD optimizer;
    MSELoss loss_fn;

    // Train for 10 epochs
    for(int i=0;i<10;i++)
    {
        Tensor<float> out = myPipeline.forward(a);
        cout<<"Loss at epoch "<<i<<": "<<loss_fn.loss(out,actual)<<endl;
        myPipeline.backward(&optimizer,&loss_fn,actual);
    }
}
```


### Create your own custom layer

```
#ifndef MyModel_H
#define MyModel_H

#include "Model.hpp"

class CustomLayer : public Model {
public:
    CustomLayer(std::pair<int,int> inputSize, int outputSize); // Add any other data you want

    // These functions are necessary for pipelines to work, make sure to implement them
    int getParamCount() override;
    std::pair<int,int> getInputSize() override;
    std::pair<int,int> getOutputSize() override;
    Tensor<float> forward(Tensor<float>) override;
    Tensor<float> OMPforward(Tensor<float>) override;

private:
    std::pair<int,int> inputSize;
    std::pair<int,int> outputSize;
    int paramCount;
    std::pair<int,int> weight_size;
};

CustomLayer::CustomLayer(std::pair<int,int> inputSize, int outputSize) : Model("Custom",true))
{
    // Call the super constructor with name of layer and a boolean definig if layer is trainable
    // Initialize weights, parameter counts etc here. Add your own initialization function if needed.
}

int Linear::getParamCount()
{
    return paramCount;
}

Tensor<float> Linear::forward(Tensor<float> input)
{
    // Define your own forward function here
}

Tensor<float> Linear::OMPforward(Tensor<float> input)
{
    // Define your own Open MP boosted forward function. Optional
}


std::pair<int,int> Linear::getOutputSize()
{
    return outputSize;
}

std::pair<int,int> Linear::getInputSize()
{
    return inputSize;
}


#endif

```

Please note that you could also override the backward function if you need to

```
void backward(Tensor<float> gradient, bool local) override;

// local to define whether its last layer or not (False for last layer, true otherwise)
// Define how gradients will be stored and updated accordingly
```

