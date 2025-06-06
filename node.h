#ifndef NODE_H
#define NODE_H

#include "tensor.h"
#include <vector>

class Node {
public:
    std::vector<tensor> operands;
    virtual ~Node() = default;
    virtual void _backward(Tensor external_grad) = 0;
};

class AddBackward : public Node {
public:
    AddBackward();
    AddBackward(tensor a, tensor b);
    void _backward(Tensor external_grad) override;
};

#endif