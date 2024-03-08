#include <iostream>
#include "Tensor.hpp"
#include <utility>
#include <cstdlib>
#include "Linear.hpp"
#include "Flatten.hpp"
#include "Pipeline.hpp"
#include "Relu.hpp"
#include "Normalize.hpp"
#include "Loss.hpp"

using namespace std;

int main()
{
    pair<int,int> size(4,4);

    Tensor<int> a = Tensor<int>::randomTensor(size);
    
    float **data = new float*[2];
    for(int i=0;i<2;i++)
    {
        data[i] = new float[4];
        for(int j=0;j<4;j++)
        {
            data[i][j] = 1;
        }
    }

    pair<int,int> out_size(2,4);
    Tensor<float> actual = Tensor(data,out_size);
    // actual.flatten().print();

    Tensor<float> trial = actual.reshape(make_pair(4,2));
    trial.print();
    // a.print();
    // Tensor b = a.copy();
    // b.print();
    // b.print();
    // cout<<"\n";
    // a.OMPtranspose();
    // a.print();
    // cout<<endl;
    // Tensor<float> b = a.convertFloat();
    // // // b.printSize();
    // Pipeline myPipeline;
    // Linear* q = new Linear(make_pair(4,4),2);
    // Normalize* l = new Normalize(make_pair(4,4));
    // Relu* r = new Relu(make_pair(4,2));
    // // //Flatten* q = new Flatten(make_pair(4,2));
    // // // q->printWeights();
    // myPipeline.add(l);
    // myPipeline.add(q);
    // myPipeline.add(r);
    // myPipeline.printPipeline();
    // Tensor<float> out = myPipeline.forward(b);
    // out.print();

    // cout<<Loss::MSELoss(out,actual);
}
