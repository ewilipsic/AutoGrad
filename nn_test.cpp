#include "nn.h"
#include "engine.h"
#include <iostream>

class NN : public model{
    NN(){
        params = {
            Linear(4,8),
            Linear(8,8),
            Linear(8,3)
        };
    }

    tensor foward(tensor input){
        tensor x;
        x = params[0].foward(input);
        x = Relu(x);
        x = params[1].foward(x);
        x = Relu(x);
        x = params[2].foward(x);
        return x;
    }
};

int main() {

    NN model();

    return 0;
}