#include "nn.h"
#include "engine.h"
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

std::pair<std::vector<tensor>, std::vector<tensor>> load_iris_dataset(const std::string& filename) {
    std::vector<tensor> features;
    std::vector<tensor> labels;
    
    std::ifstream file(filename);
    std::string line;
    

    while (std::getline(file, line)) {
        
        std::vector<float> feature_row;
        std::string species;
        float trash;
        file>>trash;
        // Read the 4 feature columns
        for (int i = 0; i < 4; i++) {
            float f; file>>f;
            feature_row.push_back(f);
        }
        
        file>>species;
       
        
        // Create feature tensor (1x4 shape for single sample)
        tensor feature_tensor(std::make_shared<Tensor>(
            std::vector<int>({1, 4}), 
            std::vector<std::vector<float>>({feature_row}), 
            0
        ));
        
        // Convert species to one-hot encoding
        std::vector<float> label_row(3, 0.0f);
        if (species == "setosa" || species == "Iris-setosa") {
            label_row[0] = 1.0f;
        } else if (species == "versicolor" || species == "Iris-versicolor") {
            label_row[1] = 1.0f;
        } else if (species == "virginica" || species == "Iris-virginica") {
            label_row[2] = 1.0f;
        }
        
        // Create label tensor (1x3 shape for one-hot encoding)
        tensor label_tensor(std::make_shared<Tensor>(
            std::vector<int>({1, 3}), 
            std::vector<std::vector<float>>({label_row}), 
            0
        ));
        
        features.push_back(feature_tensor);
        labels.push_back(label_tensor);
     
    
    }
    
    file.close();
    return std::make_pair(features, labels);
}

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

int main() {

    IrisNetwork test = IrisNetwork();
    auto [X_data, y_data] = load_iris_dataset("iris.csv");
    
    for(int epoch = 0;epoch<60;epoch++){
        float total_loss = 0.0;
        float total_corect = 0.0;
        for(int data_idx = 0;data_idx<X_data.size() ; data_idx++){

            tensor out = test.forward(X_data[data_idx]);
            int max_idx = 0;
            for(int k = 0;k<3;k++){
                if(out[k] > out[max_idx]) max_idx = k;
            }
            int max_idx2 = 0;
            for(int k = 0;k<3;k++){
                if(y_data[data_idx][k] > y_data[data_idx][max_idx2]) max_idx2 = k;
            }
            
            
            if(max_idx2 == max_idx) total_corect++;
            tensor loss = BCEloss(out,y_data[data_idx]);
            backward(loss);
            total_loss += loss[0] + loss[1] + loss[2];

        }
        std::cout<<"epoch : "<<epoch<<std::endl;
        std::cout<<"total loss : "<<total_loss<<std::endl;
        std::cout<<"total correct : "<<total_corect<<std::endl;
        test.update(0.001);
        test.zero_grad();
    }

    return 0;
}