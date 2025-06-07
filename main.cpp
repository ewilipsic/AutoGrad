#include "tensor.h"
#include "node.h"
#include "engine.h"
#include <iostream>

int main() {

    tensor a(std::make_shared<Tensor>(std::vector<int>({3, 1}), std::vector<std::vector<float>>({{3}, {2}, {1}}), 1));
    tensor b(std::make_shared<Tensor>(std::vector<int>({3, 1}), std::vector<std::vector<float>>({{3}, {2}, {1}}), 1));
    tensor c(std::make_shared<Tensor>(std::vector<int>({3, 1}), std::vector<std::vector<float>>({{2}, {1}, {1}}), 1));
    tensor d(std::make_shared<Tensor>(std::vector<int>({3, 1}), std::vector<std::vector<float>>({{1}, {2}, {1}}), 1));

    tensor e = b + c;
    tensor f = a + c;
    tensor g = e + f;

    std::cout<<"AAA\n";
    backward(g);

    printvec(a.grad()->arr);
    printvec(b.grad()->arr);
    printvec(c.grad()->arr);
    printvec(d.grad()->arr);
    printvec(e.grad()->arr);
    printvec(f.grad()->arr);
    printvec(g.grad()->arr);

    return 0;
}
