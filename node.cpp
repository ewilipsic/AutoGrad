#include "node.h"

AddBackward::AddBackward() {}

AddBackward::AddBackward(tensor a, tensor b) {
    this->operands.push_back(a);
    this->operands.push_back(b);
}

void AddBackward::_backward(Tensor external_grad) {
    for(auto& x : operands) {
        if(x.require_grad()) {
            // MEMORY LEAK
            *(x.grad()) = *(x.grad()) + external_grad;
        }
    }
}
