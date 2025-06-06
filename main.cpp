#include "tensor.h"
#include "node.h"
#include <iostream>

int main() {

    tensor q(std::make_shared<Tensor>(std::vector<int>({3, 1}), std::vector<std::vector<float>>({{3}, {2}, {1}}), 1));
    tensor t(std::make_shared<Tensor>(std::vector<int>({3, 1}), std::vector<std::vector<float>>({{3}, {2}, {1}}), 1));

    tensor k = q + t;

    return 0;
}
