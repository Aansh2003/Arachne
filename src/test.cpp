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
    // a.print();
    // Tensor b = a.copy();
    // b.print();
    // b.print();
    // cout<<"\n";
    // a.OMPtranspose();
    // a.print();
    // cout<<endl;
    Tensor<float> b = a.convertFloat();
    // // b.printSize();
    Pipeline myPipeline;
    Linear* q = new Linear(make_pair(4,4),2);
    Normalize* l = new Normalize(make_pair(4,4));
    Relu* r = new Relu(make_pair(4,2));
    // //Flatten* q = new Flatten(make_pair(4,2));
    // // q->printWeights();
    myPipeline.add(l);
    myPipeline.add(q);
    myPipeline.add(r);
    myPipeline.printPipeline();
    Tensor<float> out = myPipeline.forward(b);
    out.print();
}
