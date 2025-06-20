#ifndef NN_H
#define NN_H

#include "node.h"

namespace grail {

class module{
    public:
    virtual tensor forward(tensor input) = 0;
    virtual void update(float learning_rate) = 0;
    virtual void zero_grad() = 0;
};

class Linear : public module{
    public:
    tensor weights;
    tensor bias;
    int input_dims;
    int output_dims;

    Linear(int in_dims,int out_dims);
    tensor forward(tensor input);
    void update(float learning_rate);
    void zero_grad();
};

class Relu : public module{
    public:

    Relu();
    tensor forward(tensor input);
    void update(float learning_rate);
    void zero_grad();
};

class Sigmoid : public module{
    public:

    Sigmoid();
    tensor forward(tensor input);
    void update(float learning_rate);
    void zero_grad();
};

class Conv2D : public module{
    public:
    tensor weight;
    tensor bias;
    int stride_h;
    int stride_w;
    int padding_h;
    int padding_w;
    int kernel_h;
    int kernel_w;
    int in_channels;
    int out_channels;
    Conv2D(tensor _weight,tensor _bias,int _stride_h = 1,int _stride_w = 1,int _padding_h = 0,int _padding_w = 0,int _kernel_h = 3,int _kernel_w = 3,int _in_channels = 1,int _out_chanels = 1);
    tensor forward(tensor input);
    void update(float learning_rate);
    void zero_grad();
};

class Resize : public module{
    public:
    std::vector<int> shape;
    Resize(std::vector<int> _shape);
    tensor forward(tensor input);
    void update(float learning_rate);
    void zero_grad();
};

class model{
    public:
    std::vector<std::shared_ptr<module>> params;
    
    virtual tensor forward(tensor input) = 0;
    
    void update(float learning_rate) {
        for (auto& p : params) {
            p->update(learning_rate);
        }
    }

    void zero_grad(){
        for (auto& p : params) {
            p->zero_grad();
        }
    }
};

tensor MSEloss(const tensor& y_pred,const tensor& y_true);
tensor BCEloss(const tensor& y_pred,const tensor& y_true);

}

#endif