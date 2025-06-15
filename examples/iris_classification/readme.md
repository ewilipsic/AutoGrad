## Overview
The iris classification example demonstrates how to implement a complete neural network for multi-class classification using Grail's reverse-mode automatic differentiation capabilities . This classic machine learning problem involves categorizing iris flowers into three species based on four morphological measurements, making it an ideal benchmark for educational purposes and algorithm validation 

### Instantiating Model

    class IrisNetwork : public model {
    public:
        IrisNetwork() {
            // Create modules as shared_ptr objects
            params = {
                std::make_shared<Linear>(4, 8),
                std::make_shared<Relu>(),
                std::make_shared<Linear>(8, 8),
                std::make_shared<Relu>(),
                std::make_shared<Linear>(8, 3),
                std::make_shared<Sigmoid>(),
            };
        }
    
        tensor forward(tensor input) override {
            tensor x = input;
            for (auto& layer : params) {
                x = layer->forward(x);
            }
            return x;
        }
    };

Declare your moodel as a child of the model class and Create params variable containing all the Layers of the network. And overload foward to provide the output.

### Loss calculation and backward pass

    tensor loss = BCEloss(out,y_data[data_idx]);
    backward(loss);

loss can be calculated using Existing loss_fn provided by the library.Or Custom loss fuction can be implemented by compossing existing operations.<br>
Source Code of BCEloss as a example:<br>

    tensor BCEloss(const tensor& y_pred,const tensor& y_true){
        return -y_pred.shape()[0] * (y_true * (Log(y_pred)) + (1 - y_true) * (Log(1 - y_pred)));
    }

gradients are wrt to tensor loss are calculated using call to backward(loss) and updates can be carried out using update method provided by model

    model.update(learning_rate)


