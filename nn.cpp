#include "nn.h"

Linear::Linear(int in_dims,int out_dims){
    input_dims = in_dims;
    output_dims = out_dims;
    weights = tensor(std::vector<int>({input_dims,output_dims}),1.0,1);
    bias = tensor(std::vector<int>({1,output_dims}),1.0,1);
    weights.fill_random();
    bias.fill_random();
}

tensor Linear::forward(tensor input){
    return matmul(input,weights) + bias;
}

void Linear::update(float learning_rate){
    for(int i = 0;i<weights.arr().size();i++){
        weights[i] = weights[i] - learning_rate * (*weights.grad())[i];
    }

    for(int i = 0;i<bias.arr().size();i++){
        bias[i] = bias[i] - learning_rate * (*bias.grad())[i];
    }
}

void Linear::zero_grad(){
    weights.grad()->fill(0.0);
    bias.grad()->fill(0.0);
}

Relu::Relu(){}
tensor Relu::forward(tensor input){return relu(input);}
void Relu::update(float learning_rate){}
void Relu::zero_grad(){}

Sigmoid::Sigmoid(){}
tensor Sigmoid::forward(tensor input){return sigmoid(input);}
void Sigmoid::update(float learning_rate){}
void Sigmoid::zero_grad(){}





