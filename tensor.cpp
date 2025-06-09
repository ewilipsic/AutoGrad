#include "tensor.h"
#include "node.h"
#include <iostream>

// Tensor implementation
template <typename T1>
Tensor::Tensor(std::vector<int> _shape, const std::vector<T1>& _vec, int _require_grad) {
    int size = 1;
    for(int s : _shape) {
        size *= s;
    }
    
    if(size != get_element_count(_vec)) throw 100;
    std::cout << "Tensor Created\n";

    arr.reserve(size);
    copy_elements(_vec, arr);
    for(int x : _shape) shape.push_back(x);
    compute_strides();
    
    require_grad = _require_grad;
    if(_require_grad) {
        grad = new Tensor(shape, 0.0, 0);
    }
    out_degree = 0;
}

Tensor::Tensor(std::vector<int> _shape, float fill, int _require_grad) {
    int size = 1;
    for(int s : _shape) {
        size *= s;
        shape.push_back(s);
    }
    std::cout << "Tensor Created\n";
    
    compute_strides();
    arr = std::vector<float>(size, fill);
    
    require_grad = _require_grad;
    if(_require_grad) {
        grad = new Tensor(shape, 0.0, 0);
    }
    out_degree = 0;
}

Tensor::~Tensor() {
    std::cout << "Tensor Destroyed\n";
    if(grad) delete grad;
    if(grad_fn) delete grad_fn;
}

void Tensor::_backward(){
    if(grad_fn) grad_fn->_backward(*grad);
}

void Tensor::fill(float f){
    for(float& i : arr){
        i = f;
    }
}

void Tensor::compute_strides() {
    strides.resize(shape.size());
    if (shape.empty()) return;
    
    strides[shape.size() - 1] = 1;
    for (int i = shape.size() - 2; i >= 0; --i) {
        strides[i] = strides[i + 1] * shape[i + 1];
    }
}
    
size_t Tensor::compute_flat_index(std::initializer_list<int> indices) const {
    size_t flat_index = 0;
    auto it = indices.begin();
    for (size_t i = 0; i < indices.size() && i < strides.size(); ++i, ++it) {
        flat_index += (*it) * strides[i];
    }
    return flat_index;
}
// TensorProxy implementation
TensorProxy::TensorProxy() {}

TensorProxy::TensorProxy(std::shared_ptr<Tensor> input_ptr) : ptr(input_ptr) {}

TensorProxy::TensorProxy(std::vector<int> _shape, float fill, int _require_grad) {
    ptr = std::make_shared<Tensor>(_shape, fill, _require_grad);
}

std::vector<int>& TensorProxy::shape() const { return ptr->shape; }
std::vector<float>& TensorProxy::arr() const { return ptr->arr; }
int& TensorProxy::require_grad() const { return ptr->require_grad; }
Tensor* TensorProxy::grad() { return ptr->grad; }
Node** TensorProxy::grad_fn() { return &(ptr->grad_fn); }
int& TensorProxy::out_degree() const { return ptr->out_degree; }
void TensorProxy::_backward() const {ptr->_backward();}
void TensorProxy::fill(float f) { ptr->fill(f); }

// Explicit template instantiation
template Tensor::Tensor(std::vector<int>, const std::vector<float>&, int);
template Tensor::Tensor(std::vector<int>, const std::vector<std::vector<float>>&, int);