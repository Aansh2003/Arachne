#include <iostream>
#include "Tensor.hpp"
#include <utility>
#include <cstdlib>
#include "Linear.hpp"
#include "Flatten.hpp"
#include "Pipeline.hpp"
#include "Relu.hpp"
#include "Normalize.hpp"

using namespace std;

int main()
{
    int **arr;
    pair<int,int> size(4,4);

    arr = new int*[size.first];
    for (int i = 0; i < size.first; ++i) {
        arr[i] = new int[size.second];
    }

    for(int i=0;i<size.first;i++)
    {
        for(int j=0;j<size.second;j++)
        {
            arr[i][j] = rand()-rand();
        }
    }

    Tensor<int> a(arr,size);
    // a.print();
    // Tensor b = a.copy();
    // b.print();
    // b.print();
    // cout<<"\n";
    // a.OMPtranspose();
    // a.print();
    // cout<<endl;
    Tensor b = a.convertFloat().scalarMultiply(1.2);
    // b.printSize();
    Pipeline myPipeline;
    Linear* q = new Linear(make_pair(4,4),2);
    Normalize* l = new Normalize(make_pair(4,4));
    Relu* r = new Relu(make_pair(4,2));
    //Flatten* q = new Flatten(make_pair(4,2));
    // q->printWeights();
    // myPipeline.add(l);
    myPipeline.add(q);
    myPipeline.add(r);
    myPipeline.printPipeline();
    Tensor<float> out = myPipeline.forward(b);
    out.print();
}
