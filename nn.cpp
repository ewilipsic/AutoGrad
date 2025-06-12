#include "nn.h"

Linear::Linear(int in_dims,int out_dims){
    in_dims = input_dims;
    out_dims = output_dims;
    weights = tensor(std::vector<int>({input_dims,output_dims}),1.0,1);
}

tensor Linear::foward(tensor input){
    return matmul(input,weights);
}

void Linear::update(float learning_rate){
    for(int i = 0;i<weights.arr().size();i++){
        weights[i] = weights[i] - learning_rate * (*weights.grad())[i];
    }
}





