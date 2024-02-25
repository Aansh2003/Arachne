#include <iostream>
#include "Tensor.hpp"
#include <utility>
#include <cstdlib>
#include "Linear.hpp"
#include "Pipeline.hpp"
#include "Activation.hpp"

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
    // b.print();
    // b.map(Activation::relu);
    // b.print();
    Pipeline myPipeline;
    Linear* l = new Linear(make_pair(1,1),make_pair(2,2),"relu");
    l->printWeights();
    Linear* q = new Linear(make_pair(2,2),make_pair(2,2),"relu");
    q->printWeights();
    myPipeline.add(l);
    myPipeline.add(q);
    // myPipeline.printPipeline();
}