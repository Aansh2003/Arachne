#include <iostream>
#include "Tensor.hpp"
#include <utility>
#include <cstdlib>
#include "Linear.hpp"
#include "Flatten.hpp"
#include "Pipeline.hpp"
#include "Relu.hpp"
#include "Normalize.hpp"
#include "MSELoss.hpp"
#include "Adam.hpp"
#include <chrono>
#include "SGD.hpp"
#include "RMSProp.hpp"


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

    // a.print();
    // Tensor b = a.copy();
    // b.print();
    // b.print();
    // cout<<"\n";
    // a.OMPtranspose();
    // a.print();
    // cout<<endl;
    // // // b.printSize();
    Pipeline myPipeline;
    Linear* q = new Linear(make_pair(4,4),2);
    Normalize* l = new Normalize(make_pair(4,4));
    Relu* r = new Relu(make_pair(4,2));
    Linear* d = new Linear(make_pair(4,2),2);
    Relu *e = new Relu(make_pair(2,2),"leaky");

    MSELoss loss_fn;
    // // //Flatten* q = new Flatten(make_pair(4,2));
    // // // q->printWeights();
    myPipeline.add(l);
    myPipeline.add(q);
    myPipeline.add(r);
    myPipeline.add(d);
    myPipeline.add(e);
    myPipeline.printPipeline();
    RMSProp optimizer;

    auto start = chrono::high_resolution_clock::now();

    for(int i=0;i<10;i++)
    {
        Tensor<float> out = myPipeline.forward(a);
        // out.print();
        cout<<"Loss at epoch "<<i<<": "<<loss_fn.loss(out,actual)<<endl;

        myPipeline.backward(&optimizer,&loss_fn,actual);
    }
    auto stop = chrono::high_resolution_clock::now();
    auto duration = chrono::duration_cast<chrono::microseconds>(stop - start);
    // cout << "\nTime taken for 1000 epochs : "<< duration.count()/1000 << " milliseconds\n";
}
