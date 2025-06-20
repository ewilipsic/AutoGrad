#include "nn.h"

namespace grail{

tensor MSEloss(const tensor& y_pred,const tensor& y_true){
    return square(y_pred - y_true) / y_pred.shape()[0];
}

tensor BCEloss(const tensor& y_pred,const tensor& y_true){
    return -y_pred.shape()[0] * (y_true * (Log(y_pred)) + (1 - y_true) * (Log(1 - y_pred)));
}

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

Resize::Resize(std::vector<int> _shape){shape = _shape;}
tensor Resize::forward(tensor input){return resize(input,shape);}
void Resize::update(float learning_rate){}
void Resize::zero_grad(){}

Sigmoid::Sigmoid(){}
tensor Sigmoid::forward(tensor input){return sigmoid(input);}
void Sigmoid::update(float learning_rate){}
void Sigmoid::zero_grad(){}

Conv2D::Conv2D(tensor _weight,tensor _bias,int _stride_h,int _stride_w,int _padding_h,int _padding_w,int _kernel_h,int _kernel_w,int _in_channels,int _out_chanels){

    weight = _weight;
    bias = _bias;
    stride_h = _stride_h;
    stride_w = _stride_w;
    padding_h = _padding_h;
    padding_w = _padding_w;
    kernel_h = _kernel_h;
    kernel_w = _kernel_w;
    in_channels = _in_channels;
    out_channels = _out_chanels;

}

tensor Conv2D::forward(tensor input){
    tensor output = conv2d(input,weight,stride_h,stride_w,padding_h,padding_w);
    output = output + bias;
    return output;
}

void Conv2D::update(float learning_rate){

    for(int i = 0;i<weight.arr().size() && weight.require_grad();i++){
        weight[i] = weight[i] - learning_rate * (*weight.grad())[i];
    }

    for(int i = 0;i<bias.arr().size() && bias.require_grad();i++){
        bias[i] = bias[i] - learning_rate * (*bias.grad())[i];
    }
}

void Conv2D::zero_grad(){
    if(weight.require_grad()) weight.grad()->fill(0.0);
    if(bias.require_grad()) bias.grad()->fill(0.0);
}

}




