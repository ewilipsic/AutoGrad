#ifndef NN_H
#define NN_H

#include "node.h"

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

#endif