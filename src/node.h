#ifndef NODE_H
#define NODE_H

#include "tensor.h"
#include <vector>

namespace grail{
class Node {
public:
    std::vector<tensor> operands;
    virtual void _backward(Tensor external_grad) = 0;
    ~Node(){
       
    }
};

class AddBackward : public Node {
public:
    AddBackward();
    AddBackward(tensor a, tensor b);
    void _backward(Tensor external_grad) override;
};

class SubtractBackward : public Node {
public:
    SubtractBackward();
    SubtractBackward(tensor a, tensor b);
    void _backward(Tensor external_grad) override;
};

class MatmulBackward : public Node {
public:
    MatmulBackward();
    MatmulBackward(tensor a, tensor b);
    void _backward(Tensor external_grad) override;
};

class ScalarMulBackward : public Node {
public:
    float scalar;
    ScalarMulBackward();
    ScalarMulBackward(float _scalar, tensor b);
    void _backward(Tensor external_grad) override;
};


class ReluBackward : public Node {
public:
    ReluBackward();
    ReluBackward(tensor a);
    void _backward(Tensor external_grad) override;
};

class SquareBackward : public Node {
public:
    SquareBackward();
    SquareBackward(tensor a);
    void _backward(Tensor external_grad) override;
};

class DivisionBackward : public Node {
public:
    float scalar;
    DivisionBackward();
    DivisionBackward(float _scalar,tensor a);
    void _backward(Tensor external_grad) override;
};

class SqrtBackward : public Node {
public:
    SqrtBackward();
    SqrtBackward(tensor a);
    void _backward(Tensor external_grad) override;
};

class SigmoidBackward : public Node {
public:
    SigmoidBackward();
    SigmoidBackward(tensor a);
    void _backward(Tensor external_grad) override;
};

class LogBackward : public Node {
public:
    LogBackward();
    LogBackward(tensor a);
    void _backward(Tensor external_grad) override;
};

class ElementwiseMultBackward : public Node {
public:
    ElementwiseMultBackward();
    ElementwiseMultBackward(tensor a,tensor b);
    void _backward(Tensor external_grad) override;
};

class ScalarSubBackward : public Node {
public:
    float scalar;
    ScalarSubBackward();
    ScalarSubBackward(float _scalar,tensor b);
    void _backward(Tensor external_grad) override;
};

class conv2dBackward : public Node {
public:
    int stride_h,stride_w,padding_h,padding_w;
    conv2dBackward();
    conv2dBackward( tensor input, 
                    tensor weight, 
                    int _stride_h,int _stride_w, int _padding_h,int _padding_w);
    void _backward(Tensor external_grad) override;
};

class resizeBackward : public Node {
public:
    resizeBackward();
    resizeBackward(tensor a);
    void _backward(Tensor external_grad) override;
};

}

#endif