#ifndef Tensor_H
#define Tensor_H

#include <utility>
#include "Tensor.hpp"
#include <utility>
#include <iostream>
#include <stdexcept> 
#include <omp.h>
#include <cmath>

using namespace std;

template<typename T>
class Tensor {
public:
    // Constructor
    Tensor(T**, pair<int,int>);
    // Matrix operations
    Tensor<T> multiply(Tensor<T>);
    Tensor<T> OMPmultiply(Tensor<T>);

    Tensor<T> scalarMultiply(float);
    Tensor<T> OMPscalarMultiply(float);

    Tensor<T> add(Tensor<T>);
    Tensor<T> OMPadd(Tensor<T>);

    Tensor<float> convertFloat();
    Tensor<T> flatten();

    ~Tensor();
    
    T max();
    T min();

    // Matrix tranformations
    void transpose();
    void OMPtranspose();

    void inverse();
    void OMPinverse();

    void map(void (*func)(T*));

    // Copy function
    Tensor<T> copy();

    // Helper functions
    pair<int,int> getSize();
    void printSize();
    void print();
    T** data;
    pair<int,int> size;
private:
    // static void swap(T*,T*);
};

template<typename T>
Tensor<T>::~Tensor()
{
    for(int i=0;i<this->size.second;i++)
    {
        delete this->data[i];
    }
    delete this->data;
}

template<typename T>
Tensor<T>::Tensor(T** data, pair<int,int> size) {
    // Initialize data and size here if needed
    this->size = size;
    this->data = new T*[size.first];
    for (int i = 0; i < size.first; ++i) {
        this->data[i] = new T[size.second];
    }

    for(int i=0;i<size.first;i++)
    {
        for(int j=0;j<size.second;j++)
        {
            this->data[i][j] = *(*(data + i) + j);
        }
    }
}

template<typename T>
void Tensor<T>::printSize()
{
    cout<<"("<<size.first<<","<<size.second<<")";
}

template<typename T>
pair<int,int> Tensor<T>::getSize()
{
    return this->size;
}

template<typename T>
void Tensor<T>::print()
{
    for(int i=0;i<size.first;i++)
    {
        for(int j=0;j<size.second;j++)
        {
            cout<<this->data[i][j];
            if(j!=size.second-1)
                cout<<",";
        }
        cout<<endl;
    }
}

template<typename T>
void Tensor<T>::OMPtranspose()
{
    this->size = make_pair(this->size.second,this->size.first);
    T** temp_data = new T*[this->size.first];

    for (int i = 0; i < this->size.first; ++i) {
        temp_data[i] = new T[this->size.second];
    }

    #pragma omp parallel for collapse(2)
    for(int i=0;i<this->size.second;i++)
    {
        for(int j=0;j<this->size.first;j++)
        {
            temp_data[j][i] = this->data[i][j];
        }
    }

    T** to_delete = this->data;
    for(int i=0;i<this->size.second;i++)
    {
        delete to_delete[i];
    }
    delete to_delete;

    this->data = temp_data;
}

template<typename T>
void Tensor<T>::transpose()
{
    this->size = make_pair(this->size.second,this->size.first);
    T** temp_data = new T*[this->size.first];

    for (int i = 0; i < this->size.first; ++i) {
        temp_data[i] = new T[this->size.second];
    }

    for(int i=0;i<this->size.second;i++)
    {
        for(int j=0;j<this->size.first;j++)
        {
            temp_data[j][i] = this->data[i][j]; 
        }
    }

    T** to_delete = this->data;
    for(int i=0;i<this->size.second;i++)
    {
        delete to_delete[i];
    }
    delete to_delete;

    this->data = temp_data;
}

template<typename T>
Tensor<T> Tensor<T>::multiply(Tensor<T> b)
{
    if(this->size.second != b.size.first)
        throw std::invalid_argument("Incorrect size for matrix multiplication, must be of type - (a,b)x(b,c)");

    T** multiplied = new T*[this->size.first];
    pair<int,int> ml_size = make_pair(this->size.first,b.size.second);

    for(int i=0;i<this->size.first;i++)
    {
        multiplied[i] = new T[b.size.second];
        for(int j=0;j<b.size.second;j++)
        {
            multiplied[i][j] = 0;
        }
    }

    for (int i = 0; i < this->size.first; i++)
    {
        for (int j = 0; j < b.size.second; j++)
        {
            for (int k = 0; k < b.size.first; k++)
            {
                multiplied[i][j] += this->data[i][k] * b.data[k][j];
            }
        }
    }

    Tensor<T> output(multiplied,ml_size);

    for(int i=0;i<this->size.first;i++)
    {
        delete multiplied[i];
    }
    delete multiplied;
    
    return output;
}

template<typename T>
Tensor<T> Tensor<T>::OMPmultiply(Tensor<T> b)
{
    if(this->size.second != b.size.first)
        throw std::invalid_argument("Incorrect size for matrix multiplication, must be of type - (a,b)x(b,c)");

    T** multiplied = new T*[this->size.first];
    pair<int,int> ml_size = make_pair(this->size.first,b.size.second);

    for(int i=0;i<this->size.first;i++)
    {
        multiplied[i] = new T[b.size.second];
        for(int j=0;j<b.size.second;j++)
        {
            multiplied[i][j] = 0;
        }
    }

    #pragma omp parallel for collapse(3)
    for (int i = 0; i < this->size.first; i++)
    {
        for (int j = 0; j < b.size.second; j++)
        {
            for (int k = 0; k < b.size.first; k++)
            {
                multiplied[i][j] += this->data[i][k] * b.data[k][j];
            }
        }
    }

    Tensor<T> output(multiplied,ml_size);

    for(int i=0;i<this->size.first;i++)
    {
        delete multiplied[i];
    }
    delete multiplied;
    
    return output;
}

template<typename T>
Tensor<T> Tensor<T>::scalarMultiply(float multiplicand)
{
    T** multiplied = new T*[this->size.first];
    pair<int,int> ml_size = make_pair(this->size.first,this->size.second);
    for(int i=0;i<this->size.first;i++)
    {
        multiplied[i] = new T[this->size.second];
    }

    for(int i=0;i<this->size.first;i++)
    {
        for(int j=0;j<this->size.second;j++)
        {
            multiplied[i][j] = this->data[i][j]*multiplicand; 
        }
    }

    Tensor<T> output(multiplied,ml_size);

    for(int i=0;i<this->size.first;i++)
    {
        delete multiplied[i];
    }
    delete multiplied;
    
    return output;
}

template<typename T>
Tensor<T> Tensor<T>::OMPscalarMultiply(float multiplicand)
{
    T** multiplied = new T*[this->size.first];
    pair<int,int> ml_size = make_pair(this->size.first,this->size.second);
    for(int i=0;i<this->size.first;i++)
    {
        multiplied[i] = new T[this->size.second];
    }

    #pragma omp parallel for collapse(2)
    for(int i=0;i<this->size.first;i++)
    {
        for(int j=0;j<this->size.second;j++)
        {
            multiplied[i][j] = this->data[i][j]*multiplicand; 
        }
    }

    Tensor<T> output(multiplied,ml_size);

    for(int i=0;i<this->size.first;i++)
    {
        delete multiplied[i];
    }
    delete multiplied;
    
    return output;
}

template<typename T>
Tensor<T> Tensor<T>::add(Tensor<T> adder)
{
    if(this->size.first != adder.size.first || this->size.second != adder.size.second)
        throw std::invalid_argument("Matrices should be of same dimension (a,b) + (a,b)");

    T** added = new T*[this->size.first];
    pair<int,int> ml_size = make_pair(this->size.first,this->size.second);
    for(int i=0;i<this->size.first;i++)
    {
        added[i] = new int[this->size.second];
    }

    for(int i=0;i<this->size.first;i++)
    {
        for(int j=0;j<this->size.second;j++)
        {
            added[i][j] = this->data[i][j]+adder.data[i][j]; 
        }
    }

    Tensor<T> output(added,ml_size);

    for(int i=0;i<this->size.first;i++)
    {
        delete added[i];
    }
    delete added;
    
    return output;
}

template<typename T>
Tensor<T> Tensor<T>::OMPadd(Tensor<T> adder)
{
    if(this->size.first != adder.size.first || this->size.second != adder.size.second)
        throw std::invalid_argument("Matrices should be of same dimension (a,b) + (a,b)");

    T** added = new T*[this->size.first];
    pair<int,int> ml_size = make_pair(this->size.first,this->size.second);
    for(int i=0;i<this->size.first;i++)
    {
        added[i] = new int[this->size.second];
    }

    #pragma omp parallel for collapse(2)
    for(int i=0;i<this->size.first;i++)
    {
        for(int j=0;j<this->size.second;j++)
        {
            added[i][j] = this->data[i][j]+adder.data[i][j]; 
        }
    }
    Tensor<T> output(added,ml_size);

    for(int i=0;i<this->size.first;i++)
    {
        delete added[i];
    }
    delete added;

    return output;
}

template<typename T>
Tensor<T> Tensor<T>::copy()
{
    return Tensor<T>(this->data,this->size);
}
// template<typename T>
// void Tensor<T>::swap(T* a,T* b)
// {
//     T temp = *a;
//     *a = *b;
//     *b = temp;
// }


// template<typename T>
// void Tensor<T>::inverse() {
//     int m = size.first;
//     int n = size.second;

//     if(!std::is_floating_point_v<T>)
//         cerr<"Inversion may result in incorrect truncation for integer tensors. Use convertFloat() to get accurate output."

//     for(int i =0;i<m;i++)
//     {
//         if(this->data[i][i] == 0.0)
//         {
//             throw std::runtime_error("Inversion not possible for singular matrices")
//         }
//         for(int j=i+1;j<m;j++)
//         {
            
//         }
//     }

// }

template<typename T>
Tensor<float> Tensor<T>::convertFloat()
{
    float** floatTensor = new float*[this->size.first];
    for(int i=0;i<this->size.first;i++)
    {
        floatTensor[i] = new float[this->size.second];
    }

    for(int i = 0;i<this->size.first;i++)
    {
        for(int j=0;j<this->size.second;j++)
        {
            floatTensor[i][j] = float(this->data[i][j]);
        }
    }
    Tensor<float> output(floatTensor,this->size);

    for(int i=0;i<this->size.first;i++)
    {
        delete floatTensor[i];
    }
    delete floatTensor;

    return output;
}

template<typename T>
void Tensor<T>::map(void (*func)(T*))
{
    for(int i=0;i<this->size.first;i++)
    {
        for(int j=0;j<this->size.second;j++)
        {
            func(&this->data[i][j]);
        }
    }
}

template<typename T>
Tensor<T> Tensor<T>::flatten()
{
    T** data = new T*[1];
    data[0] = new T[this->size.first*this->size.second];

    for(int i=0;i<this->size.first;i++)
    {
        for(int j=0;j<this->size.second;j++)
        {
            data[0][i*j] = this->data[i][j];
        }
    }
    Tensor<T> output(data,make_pair(1,this->size.first*this->size.second));
    return output;
}

template<typename T>
T Tensor<T>::max()
{
    T small = this->data[0][0];
    for(int i=0;i<this->size.first;i++)
    {
        for(int j=0;j<this->size.second;j++)
        {
		if(this->data[i][j] > small)
		    small = this->data[i][j];
        }
    }

    return small;
}

template<typename T>
T Tensor<T>::min()
{
    T big = this->data[0][0];
    
    for(int i=0;i<this->size.first;i++)
    {
        for(int j=0;j<this->size.second;j++)
        {
		if(this->data[i][j] < big)
		    big = this->data[i][j];
        }
    }

    return big;
}


#endif
