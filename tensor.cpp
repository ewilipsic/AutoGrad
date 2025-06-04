#include<vector>
#include<iostream>
#include<memory>

void printvec(std::vector<float> vec){
    for(auto& x : vec) std::cout<<x<<" ";
    std::cout<<std::endl;
}
void printvec(std::vector<int> vec){
    for(auto& x : vec) std::cout<<x<<" ";
    std::cout<<std::endl;
}

int get_element_count(const float f){
    return 1;
}

template<typename T>
int get_element_count(const std::vector<T>& vec){
    int ret = 0;
    for(int i = 0;i<vec.size();i++){
        ret += get_element_count(vec[i]);
    }
    return ret;
}


void copy_elements(const float f,std::vector<float>& vec){
    vec.push_back(f);
}

template<typename T1>
void copy_elements(const std::vector<T1>& f,std::vector<float>& vec){
    for(int i = 0;i<f.size();i++){
        copy_elements(f[i],vec);
    }
}


class Tensor{
    public:
    std::vector<float> arr;
    std::vector<int> shape;
    int require_grad;
    template <typename T1>
    Tensor(std::vector<int> _shape,const std::vector<T1>& _vec,int _require_grad = 1){
        std::cout<<"Tensor Created\n";
        int size = 1;
        require_grad = _require_grad;
        for(int s : _shape){
            size *= s;
        }
        if(size == get_element_count(_vec)){
            arr.reserve(size);
            copy_elements(_vec,arr);
            shape = _shape;
        }
        else{
            throw 100;
        }
    }

   
    Tensor(std::vector<int> _shape,int _require_grad = 1){
        std::cout<<"Tensor Created\n";
        int size = 1;
        require_grad = _require_grad;
        for(int s : _shape){
            size *= s;
        }
        arr = std::vector<float>(size,0);
        shape = _shape; 
    }

    ~Tensor(){
        std::cout<<"Tensor Destroyed\n";
    }
};


class TensorProxy{
public:
    std::shared_ptr<Tensor> ptr;

    TensorProxy(){}
    TensorProxy(std::shared_ptr<Tensor> input_ptr){
        ptr = input_ptr;
    }
    
    std::vector<int>& shape() const { return ptr->shape; }
    std::vector<float>& arr() const { return ptr->arr; }
    int& require_grad() const { return ptr->require_grad; }
};

using tensor = TensorProxy;

Tensor operator+(const Tensor& a,const Tensor& b){
    if(a.shape != b.shape){
        throw 200;
    }
    Tensor ret(a.shape);
    for(int i = 0;i<a.arr.size();i++){
        ret.arr[i] = a.arr[i] + b.arr[i];
    }
    return ret;
};

tensor operator+(const tensor& a, const tensor& b){
    if(a.shape() != b.shape()){
        throw 201;
    }

    tensor ret = tensor(std::make_shared<Tensor>(a.shape()));

    for(int i = 0;i<a.arr().size();i++){
        ret.arr()[i] = a.arr()[i] + b.arr()[i];
    }

    return ret;
}

Tensor operator*(const float& a,const Tensor& b){
    Tensor ret(b.shape);
    for(int i = 0;i<b.arr.size();i++){
        ret.arr[i] = a * b.arr[i];
    }
    return ret;
};

tensor operator*(const float& a,const tensor& b){
    tensor ret = tensor(std::make_shared<Tensor>(b.shape()));

    for(int i = 0;i<b.arr().size();i++){
        ret.arr()[i] = a * b.arr()[i];
    }
    return ret;
};

Tensor matmul(const Tensor& a,const Tensor& b){
    int m = a.shape[0];
    int n = a.shape[1];
    int o = b.shape[1];
    Tensor ret(std::vector<int>({m,o}));
    for(int i = 0;i<m;i++){
        for(int j = 0;j<o;j++){
            float sum = 0;
            for(int k = 0;k<n;k++){
                sum += a.arr[n*i + k] * b.arr[k*o + j];
            }
            ret.arr[i * o + j] = sum;
        }
    }
    return ret;
}

tensor matmul(const tensor& a,const tensor& b){
    int m = a.shape()[0];
    int n = a.shape()[1];
    int o = b.shape()[1];

    tensor ret = tensor(std::make_shared<Tensor>(std::vector<int>({m,o})));
    for(int i = 0;i<m;i++){
        for(int j = 0;j<o;j++){
            float sum = 0;
            for(int k = 0;k<n;k++){
                sum += a.arr()[n*i + k] * b.arr()[k*o + j];
            }
            ret.arr()[i * o + j] = sum;
        }
    }
    return ret;
}

// class Node{
//     public:
//     std::vector<std::shared_ptr<Tensor>> operands;
// };

// class AddBackward : public Node{
//     public:
//     AddBackward(tensor a,tensor b){
//         this->operands.push_back(std::make_shared<Tensor>(a));
//         this->operands.push_back(std::make_shared<Tensor>(b));
//     }

// };


int main(){
    
  

    tensor q(std::make_shared<Tensor>(std::vector<int>({3,1}),std::vector<float>({{3},{2},{1}})));
    tensor t(std::make_shared<Tensor>(std::vector<int>({1,2}),std::vector<float>({{1,3}})));

    Tensor Q = Tensor(std::vector<int>({3,1}),std::vector<float>({{3},{2},{1}}));
    Tensor T = Tensor(std::vector<int>({1,2}),std::vector<float>({{1,3}}));


    Tensor K = matmul(Q,T);
    tensor k = matmul(q,t);

    printvec(K.arr);
    printvec(k.arr());

    


}