#ifndef TENSOR_H
#define TENSOR_H

#include <vector>
#include <memory>
#include <iostream>

// Forward declarations
class Node;
class Tensor;


class Tensor{
    public:

    int out_degree = 0;
    int require_grad = 0;

    Tensor* grad = nullptr;
    Node* grad_fn = nullptr;

    std::vector<float> arr;
    std::vector<int> shape;
    std::vector<int> strides;

    template <typename T1>
    Tensor(std::vector<int> _shape,const std::vector<T1>& _vec,int _require_grad = 0);
    Tensor(std::vector<int> _shape,float fill,int _require_grad = 0);

    void compute_strides();
    size_t compute_flat_index(std::initializer_list<int> indices) const;

    void fill(float f);
    void _backward();

    float& operator[] (size_t index) {
        return arr[index];
    }
    
    // Multi-dimensional access using initializer list
    float& operator[](std::initializer_list<int> indices) {
        return arr[compute_flat_index(indices)];
    }

    ~Tensor();
};


class TensorProxy {
public:
    std::shared_ptr<Tensor> ptr;
    
    TensorProxy();
    TensorProxy(std::shared_ptr<Tensor> input_ptr);
    TensorProxy(std::vector<int> _shape, float fill, int _require_grad = 0);
    
    std::vector<int>& shape() const;
    std::vector<float>& arr() const;
    int& require_grad() const;
    Tensor* grad();
    Node** grad_fn();
    int& out_degree() const;
    void _backward() const;
    void fill(float f) ;

    float& operator[] (size_t index) {
        return (*ptr)[index];
    }

    // Multi-dimensional access using initializer list
    float& operator[](std::initializer_list<int> indices) {
        return  (*ptr)[indices];
    }
    
};

using tensor = TensorProxy;

// Function declarations
int get_element_count(const float f);

template<typename T>
int get_element_count(const std::vector<T>& vec);

void copy_elements(const float f, std::vector<float>& vec);

template<typename T1>
void copy_elements(const std::vector<T1>& f, std::vector<float>& vec);

void printvec(std::vector<float> vec);
void printvec(std::vector<int> vec);

// Operator declarations
Tensor operator+(const Tensor& a, const Tensor& b);
tensor operator+(const tensor a, const tensor b);
Tensor operator*(const float& a, const Tensor& b);
tensor operator*(const float& a, const tensor& b);
Tensor matmul(const Tensor& a, const Tensor& b);
tensor matmul(const tensor& a, const tensor& b);
tensor Relu(const tensor& a);
bool operator<(const tensor& a, const tensor& b);

#endif