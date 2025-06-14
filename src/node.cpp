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

}
