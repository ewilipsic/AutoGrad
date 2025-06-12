#ifndef NN_H
#define NN_H

#include "node.h"

class module{
    public:
    virtual  tensor foward(tensor input) = 0;
    virtual void update(float learning_rate) = 0;
};

class Linear : public module{
    public:
    tensor weights;
    int input_dims;
    int output_dims;

    Linear(int in_dims,int out_dims);
    tensor foward(tensor input);
    void update(float learning_rate);
};

class model{
    public:
    std::vector<module> params;
    virtual  tensor foward(tensor input) = 0;
    virtual void update(float learning_rate){
        for(module& p : params){
            p.update(learning_rate);
        }
    }

};

#endif