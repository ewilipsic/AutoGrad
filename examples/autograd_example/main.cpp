#include <iostream>
#include "..\..\src\grail.h" 

using namespace grail;



int main() {

    tensor a(   std::make_shared<Tensor>(std::vector<int>({3, 3}), 
                std::vector<std::vector<float>>({{2,1,2}, {2,5,6}, {1,2,2}}), 1));
    tensor b(   std::make_shared<Tensor>(std::vector<int>({3, 3}), 
                std::vector<std::vector<float>>({{1,1,1}, {1,2,3}, {1,6,1}}), 1));

    tensor c = matmul(a,b);
    tensor d = sigmoid(a) + c;
    tensor e = c + d;

    backward(e);

    std::cout<<"grad a: ";printvec(a.grad()->arr);
    std::cout<<"grad b: ";printvec(b.grad()->arr);
    std::cout<<"grad c: ";printvec(c.grad()->arr);
    std::cout<<"grad d: ";printvec(d.grad()->arr);
    std::cout<<"grad e: ";printvec(e.grad()->arr); 
   
    return 0;
}
