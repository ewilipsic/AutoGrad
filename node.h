#ifndef NODE_H
#define NODE_H

#include "tensor.h"
#include <vector>

class Node {
public:
    std::vector<tensor> operands;
    virtual void _backward(Tensor external_grad) = 0;

    ~Node(){
        std::cout<<"Node Destroyed\n";
    }
};

class AddBackward : public Node {
public:
    AddBackward();
    AddBackward(tensor a, tensor b);
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

#endif