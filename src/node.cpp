#include "node.h"

namespace grail {
AddBackward::AddBackward() {}

AddBackward::AddBackward(tensor a, tensor b) {
    this->operands.push_back(a);
    this->operands.push_back(b);
}

void AddBackward::_backward(Tensor external_grad) {
    for(auto& x : operands) {
        if(x.require_grad()) {
            for(int i = 0;i<x.arr().size();i++){
                (*(x.grad()))[i] = (*(x.grad()))[i] + external_grad[i];
            }
        }
    }
}

SubtractBackward::SubtractBackward() {}

SubtractBackward::SubtractBackward(tensor a, tensor b) {
    this->operands.push_back(a);
    this->operands.push_back(b);
}

void SubtractBackward::_backward(Tensor external_grad) {
    if(operands[0].require_grad()) {
        auto& x = operands[0];
        if(x.require_grad()) {
            for(int i = 0;i<x.arr().size();i++){
                (*(x.grad()))[i] = (*(x.grad()))[i] + external_grad[i];
            }
        }
    }
    if(operands[1].require_grad()) {
        auto& x = operands[1];
        if(x.require_grad()) {
            for(int i = 0;i<x.arr().size();i++){
                (*(x.grad()))[i] = (*(x.grad()))[i] - external_grad[i];
            }
        }
    }
}

MatmulBackward::MatmulBackward() {}

MatmulBackward::MatmulBackward(tensor a, tensor b) {
    this->operands.push_back(a);
    this->operands.push_back(b);
}

void MatmulBackward::_backward(Tensor external_grad) {
    if(operands[0].require_grad()){
        for(int i = 0;i<operands[0].shape()[0];i++){
            for(int j = 0;j<operands[0].shape()[1];j++){
                float sum = 0;
                for(int q = 0;q<operands[1].shape()[1];q++){
                    sum += external_grad[{i,q}] * (operands[1])[{j,q}];
                }
                (*(operands[0].grad()))[{i,j}] += sum;
            }
        }
    }
    if(operands[1].require_grad()){
        for(int q = 0;q<operands[1].shape()[1];q++){
            for(int j = 0;j<operands[0].shape()[1];j++){
                float sum = 0;
                for(int i = 0;i<operands[0].shape()[0];i++){
                    sum += external_grad[{i,q}] * (operands[0])[{i,j}];
                }
                (*(operands[1].grad()))[{j,q}] += sum;
            }
        }
    }
}

ScalarMulBackward::ScalarMulBackward() {}

ScalarMulBackward::ScalarMulBackward(float _scalar, tensor b) {
    this->operands.push_back(b);
    scalar = _scalar;
}

void ScalarMulBackward::_backward(Tensor external_grad) {
    for(auto& x : operands) {
        if(x.require_grad()) {
            for(int i = 0;i<x.arr().size();i++){
                (*(x.grad()))[i] = (*(x.grad()))[i] + external_grad[i] * scalar ;
            }
        }
    }
}

ReluBackward::ReluBackward() {}

ReluBackward::ReluBackward(tensor a) {
    this->operands.push_back(a);
}

void ReluBackward::_backward(Tensor external_grad) {
    for(auto& x : operands) {
        if(x.require_grad()) {
            for(int i = 0;i<(x.grad())->arr.size();i++){
                (*(x.grad()))[i] = ((x[i] > 0) ? external_grad[i] : 0.0);
            }
        }
    }
}

SquareBackward::SquareBackward() {}

SquareBackward::SquareBackward(tensor a) {
    this->operands.push_back(a);
}

void SquareBackward::_backward(Tensor external_grad) {
    for(auto& x : operands) {
        if(x.require_grad()) {
            for(int i = 0;i<x.arr().size();i++){
                (*(x.grad()))[i] = (*(x.grad()))[i] + external_grad[i] * 2 * x[i];
            }
        }
    }
}

DivisionBackward::DivisionBackward() {}

DivisionBackward::DivisionBackward(float _scalar,tensor a) {
    this->operands.push_back(a);
    scalar = _scalar;
}

void DivisionBackward::_backward(Tensor external_grad) {
    for(auto& x : operands) {
        if(x.require_grad()) {
            for(int i = 0;i<x.arr().size();i++){
                (*(x.grad()))[i] = (*(x.grad()))[i] + external_grad[i] / scalar ;
            }
        }
    }
}

SqrtBackward::SqrtBackward() {}

SqrtBackward::SqrtBackward(tensor a) {
    this->operands.push_back(a);
}

void SqrtBackward::_backward(Tensor external_grad) {
    for(auto& x : operands) {
        if(x.require_grad()) {
            for(int i = 0;i<x.arr().size();i++){
                (*(x.grad()))[i] = (*(x.grad()))[i] + external_grad[i] * 0.5 * sqrtf(x[i]) ;
            }
        }
    }
}

float sigmoid(float f){
    return 1.0 / (1 + expf(-f));
}

SigmoidBackward::SigmoidBackward() {}

SigmoidBackward::SigmoidBackward(tensor a) {
    this->operands.push_back(a);
}

void SigmoidBackward::_backward(Tensor external_grad) {
    for(auto& x : operands) {
        if(x.require_grad()) {
            for(int i = 0;i<x.arr().size();i++){
                (*(x.grad()))[i] = (*(x.grad()))[i] + external_grad[i] * sigmoid(x[i]) * (1.0 - sigmoid(x[i])) ;
            }
        }
    }
}

LogBackward::LogBackward() {}

LogBackward::LogBackward(tensor a) {
    this->operands.push_back(a);
}

void LogBackward::_backward(Tensor external_grad) {
    for(auto& x : operands) {
        if(x.require_grad()) {
            for(int i = 0;i<x.arr().size();i++){
                (*(x.grad()))[i] = (*(x.grad()))[i] + external_grad[i] * (1.0/x[i]) ;
            }
        }
    }
}

ElementwiseMultBackward::ElementwiseMultBackward() {}

ElementwiseMultBackward::ElementwiseMultBackward(tensor a, tensor b) {
    this->operands.push_back(a);
    this->operands.push_back(b);
}

void ElementwiseMultBackward::_backward(Tensor external_grad) {
    if(operands[0].require_grad()){
        for(int i = 0;i<operands[0].shape()[0];i++){
            for(int j = 0;j<operands[0].shape()[1];j++){
                (*(operands[0].grad()))[{i,j}] += external_grad[{i,j}] * (operands[1])[{i,j}];
            }
        }
    }
    if(operands[1].require_grad()){
        for(int i = 0;i<operands[1].shape()[0];i++){
            for(int j = 0;j<operands[1].shape()[1];j++){
                (*(operands[1].grad()))[{i,j}] += external_grad[{i,j}] * (operands[0])[{i,j}];
            }
        }
    }
}

ScalarSubBackward::ScalarSubBackward() {}

ScalarSubBackward::ScalarSubBackward(float _scalar, tensor b) {
    this->operands.push_back(b);
    scalar = _scalar;
}

void ScalarSubBackward::_backward(Tensor external_grad) {
    for(auto& x : operands) {
        if(x.require_grad()) {
            for(int i = 0;i<x.arr().size();i++){
                (*(x.grad()))[i] = (*(x.grad()))[i] - external_grad[i];
            }
        }
    }
}

conv2dBackward::conv2dBackward() {}

conv2dBackward::conv2dBackward( tensor input, 
                    tensor weight, 
                    int _stride_h,int _stride_w, int _padding_h,int _padding_w){

    this->operands.push_back(input);
    this->operands.push_back(weight);
    stride_h = _stride_h;
    stride_w = _stride_w;
    padding_h = _padding_h;
    padding_w = _padding_w;
}

void conv2dBackward::_backward(Tensor external_grad) {

    tensor input = operands[0];
    tensor weight = operands[1];

    int batch_size = input.shape()[0];
    int in_channels = input.shape()[1];
    int in_height = input.shape()[2];
    int in_width = input.shape()[3];

    int out_channels = weight.shape()[0];
    int kernel_height = weight.shape()[2];
    int kernel_width = weight.shape()[3];

    int out_height = (in_height + 2 * padding_h - kernel_height) / stride_h + 1;
    int out_width = (in_width + 2 * padding_w - kernel_width) / stride_w + 1;

    if(input.require_grad()){
        for(int b = 0;b<batch_size;b++){
            for(int ic = 0;ic<in_channels;ic++){
            
                for(int ih = 0;ih<in_height;ih++){
                    for(int iw = 0;iw<in_width;iw++){

                        float sum = 0.0;

                        for(int oc = 0;oc < out_channels ;oc++){
                            int ki_initial = (1 + iw + kernel_width) / stride_w + 1;
                            int kj_initial = (1 + ih + kernel_height) / stride_h + 1;


                            
                            for(int ki = ki_initial;1 + (ki_initial - ki)*stride_w <= kernel_width ;ki--){
                                for(int kj = kj_initial; 1 + (kj_initial - kj)*stride_h <= kernel_height  ; kj--){

                                    // implicit zero padding
                                    if(!(ki >= 0 && kj >= 0 && ki < out_width && kj < out_height)) continue;
                                    sum += external_grad[{b,oc,kj,ki}] * weight[{oc,ic,stride_h * (kj_initial - kj),stride_w * (ki_initial - ki)}];

                                }
                            }
                        }

                        (*input.grad())[{b,ic,ih,iw}] += sum;
                    }
                }
            }
        }
    }

    if(weight.require_grad()){
        for(int b = 0;b < batch_size;b++){
        for(int oc = 0;oc < out_channels;oc++){
            for(int ic = 0;ic < in_channels ;ic++){
                for(int kh = 0;kh < kernel_height;kh++){
                    for(int kw = 0;kw < kernel_width;kw++){

                        int start_i = -padding_w + kw;
                        int start_j = -padding_h + kh;
                        
                        float sum = 0.0;
                        
                        int output_i = 0;
                        int output_j = 0;
                        for(int input_i = start_i ; input_i < in_width ; input_i+= stride_w){
                            int output_j = 0;
                            for(int input_j = start_j ; input_j < in_height ; input_j += stride_h){
                                if(!(input_j >= 0 && input_i >= 0 && output_i < out_width && output_j < out_height)) continue;
                                sum += external_grad[{b,oc,output_j,output_i}] * input[{b,ic,input_j,input_i}];
                                output_j++; 
                            }
                            output_i++;
                        }

                        (*weight.grad())[{oc,ic,kh,kw}] += sum;
                    }
                }
            }
        }
        }
    }


    
}

resizeBackward::resizeBackward() {}

resizeBackward::resizeBackward(tensor a) {
    this->operands.push_back(a);
}

void resizeBackward::_backward(Tensor external_grad) {
    for(auto& x : operands) {
        if(x.require_grad()) {
            for(int i = 0;i<(x.grad())->arr.size();i++){
                (*(x.grad()))[i] = external_grad[i];
            }
        }
    }
}


}


