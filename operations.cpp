#include "tensor.h"
#include "node.h"

Tensor operator+(const Tensor& a, const Tensor& b) {
    if(a.shape != b.shape) {
        throw 200;
    }
    Tensor ret(a.shape, 0, 0);
    for(int i = 0; i < a.arr.size(); i++) {
        ret.arr[i] = a.arr[i] + b.arr[i];
    }
    return ret;
}

tensor operator+(const tensor a, const tensor b) {
    bool get_backward = false;
    if(a.require_grad() || b.require_grad()) get_backward = true;
    
    if(a.shape() != b.shape()) {
        throw 201;
    }
    
    tensor ret = tensor(a.shape(), 0.0, get_backward);
    
    for(int i = 0; i < a.arr().size(); i++) {
        ret.arr()[i] = a.arr()[i] + b.arr()[i];
    }
    
    if(get_backward) {
        auto back_fn = new AddBackward(a, b);
        *(ret.grad_fn()) = back_fn;
    }
    
    return ret;
}

tensor operator-(const tensor& a, const tensor& b) {
    bool get_backward = false;
    if(a.require_grad() || b.require_grad()) get_backward = true;
    
    if(a.shape() != b.shape()) {
        throw 201;
    }
    
    tensor ret = tensor(a.shape(), 0.0, get_backward);
    
    for(int i = 0; i < a.arr().size(); i++) {
        ret.arr()[i] = a.arr()[i] - b.arr()[i];
    }
    
    if(get_backward) {
        auto back_fn = new SubtractBackward(a, b);
        *(ret.grad_fn()) = back_fn;
    }
    
    return ret;
}

Tensor operator*(const float& a, const Tensor& b) {
    Tensor ret(b.shape, 0, 0);
    for(int i = 0; i < b.arr.size(); i++) {
        ret.arr[i] = a * b.arr[i];
    }
    return ret;
}


tensor operator*(const float& a, const tensor& b) {
    bool get_backward = false;
    if(b.require_grad()) get_backward = true;

    tensor ret = tensor(b.shape(), 0.0, get_backward);
    
    for(int i = 0; i < b.arr().size(); i++) {
        ret.arr()[i] = a * b.arr()[i];
    }

    if(get_backward) {
        auto back_fn = new ScalarMulBackward(a, b);
        *(ret.grad_fn()) = back_fn;
    }
    return ret;
}

tensor operator-(const float& a, const tensor& b) {
    bool get_backward = false;
    if(b.require_grad()) get_backward = true;

    tensor ret = tensor(b.shape(), 0.0, get_backward);
    
    for(int i = 0; i < b.arr().size(); i++) {
        ret.arr()[i] = a - b.arr()[i];
    }

    if(get_backward) {
        auto back_fn = new ScalarSubBackward(a, b);
        *(ret.grad_fn()) = back_fn;
    }
    return ret;
}

Tensor matmul(const Tensor& a, const Tensor& b) {
    int m = a.shape[0];
    int n = a.shape[1];
    int o = b.shape[1];
    Tensor ret(std::vector<int>({m, o}), 0);
    for(int i = 0; i < m; i++) {
        for(int j = 0; j < o; j++) {
            float sum = 0;
            for(int k = 0; k < n; k++) {
                sum += a.arr[n*i + k] * b.arr[k*o + j];
            }
            ret.arr[i * o + j] = sum;
        }
    }
    return ret;
}

tensor matmul(const tensor& a, const tensor& b) {
    bool get_backward = false;
    if(a.require_grad() || b.require_grad()) get_backward = true;
    int m = a.shape()[0];
    int n = a.shape()[1];
    int o = b.shape()[1];
    
    tensor ret = tensor(std::vector<int>({m, o}), 0.0, get_backward);
    for(int i = 0; i < m; i++) {
        for(int j = 0; j < o; j++) {
            float sum = 0;
            for(int k = 0; k < n; k++) {
                sum += a.arr()[n*i + k] * b.arr()[k*o + j];
            }
            ret.arr()[i * o + j] = sum;
        }
    }

    if(get_backward) {
        auto back_fn = new MatmulBackward(a, b);
        *(ret.grad_fn()) = back_fn;
    }

    return ret;
}

tensor relu(const tensor& a){
    bool get_backward = false;
    if(a.require_grad()) get_backward = true;
    
    tensor ret = tensor(a.shape(), 0.0, get_backward);
    
    for(int i = 0; i < a.arr().size(); i++) {
        ret.arr()[i] = ((a.arr()[i] > 0) ? a.arr()[i] : 0.0);
    }
    
    if(get_backward) {
        auto back_fn = new ReluBackward(a);
        *(ret.grad_fn()) = back_fn;
    }
    
    return ret;

}

bool operator<(const tensor& a, const tensor& b){
    return a.ptr.get() < b.ptr.get();
}

tensor square(const tensor& a){
    bool get_backward = false;
    if(a.require_grad()) get_backward = true;
    
    tensor ret = tensor(a.shape(), 0.0, get_backward);
    
    for(int i = 0; i < a.arr().size(); i++) {
        ret.arr()[i] = a.arr()[i] * a.arr()[i];
    }
    
    if(get_backward) {
        auto back_fn = new SquareBackward(a);
        *(ret.grad_fn()) = back_fn;
    }
    
    return ret;
}

tensor sqrt(const tensor& a){
    bool get_backward = false;
    if(a.require_grad()) get_backward = true;
    
    tensor ret = tensor(a.shape(), 0.0, get_backward);
    
    for(int i = 0; i < a.arr().size(); i++) {
        ret.arr()[i] = sqrtf(a.arr()[i]);
    }
    
    if(get_backward) {
        auto back_fn = new SqrtBackward(a);
        *(ret.grad_fn()) = back_fn;
    }
    
    return ret;
}

tensor sigmoid(const tensor& a){
    bool get_backward = false;
    if(a.require_grad()) get_backward = true;
    
    tensor ret = tensor(a.shape(), 0.0, get_backward);
    
    for(int i = 0; i < a.arr().size(); i++) {
        ret.arr()[i] = 1.0 / (1 + expf(-a.arr()[i]));
    }
    
    if(get_backward) {
        auto back_fn = new SigmoidBackward(a);
        *(ret.grad_fn()) = back_fn;
    }
    
    return ret;
}

tensor operator/(const tensor& b,const float& a){
    bool get_backward = false;
    if(b.require_grad()) get_backward = true;

    tensor ret = tensor(b.shape(), 0.0, get_backward);
    
    for(int i = 0; i < b.arr().size(); i++) {
        ret.arr()[i] = b.arr()[i] / a;
    }

    if(get_backward) {
        auto back_fn = new DivisionBackward(a, b);
        *(ret.grad_fn()) = back_fn;
    }
    return ret;
}

tensor Log(const tensor& a){
    bool get_backward = false;
    if(a.require_grad()) get_backward = true;

    tensor ret = tensor(a.shape(), 0.0, get_backward);
    
    for(int i = 0; i < a.arr().size(); i++) {
        ret.arr()[i] = log(a.arr()[i]);
    }

    if(get_backward) {
        auto back_fn = new LogBackward(a);
        *(ret.grad_fn()) = back_fn;
    }
    return ret;
}

tensor operator*(const tensor& a, const tensor& b) {
    bool get_backward = false;
    if(a.require_grad() || b.require_grad()) get_backward = true;
    
    if(a.shape() != b.shape()) {
        throw 201;
    }
    
    tensor ret = tensor(a.shape(), 0.0, get_backward);
    
    for(int i = 0; i < a.arr().size(); i++) {
        ret.arr()[i] = a.arr()[i] * b.arr()[i];
    }
    
    if(get_backward) {
        auto back_fn = new ElementwiseMultBackward(a, b);
        *(ret.grad_fn()) = back_fn;
    }
    
    return ret;
}