#include "tensor.h"
#include "node.h"
#include "engine.h"
#include <iostream>

int main() {

    tensor a(std::make_shared<Tensor>(std::vector<int>({3, 3}), std::vector<std::vector<float>>({{-1,-1,2}, {2,-3,4}, {2,-3,4}}), 1));
    tensor b(std::make_shared<Tensor>(std::vector<int>({3, 3}), std::vector<std::vector<float>>({{-1,1,2}, {2,3,4}, {9,1,4}}), 1));
    
    tensor c = Relu(a);

    tensor d = matmul(b,c);

    backward(d);

    printvec((a.grad())->arr);
    printvec((b.grad())->arr);
    printvec((c.grad())->arr);
    printvec((d.grad())->arr);
    

    

    return 0;
}
