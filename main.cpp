#include "tensor.h"
#include "node.h"
#include "engine.h"
#include <iostream>

int main() {

    tensor a(std::make_shared<Tensor>(std::vector<int>({3, 3}), std::vector<std::vector<float>>({{2,2,2}, {2,2,2}, {2,2,2}}), 1));
    tensor b(std::make_shared<Tensor>(std::vector<int>({3, 3}), std::vector<std::vector<float>>({{1,1,1}, {1,1,1}, {1,1,1}}), 1));
    tensor c = sigmoid(b);
    tensor d = sigmoid(a);
    tensor e = c + d;

    backward(e);

    printvec(a.grad()->arr);
    printvec(b.grad()->arr);
    printvec(c.grad()->arr);
    printvec(d.grad()->arr);
    printvec(e.grad()->arr); 
   
    return 0;
}
