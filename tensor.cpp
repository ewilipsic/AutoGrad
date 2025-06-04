#include<vector>
#include<iostream>

template<typename T>
void printvec(std::vector<T> vec){
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

template<typename T>
void copy_elements(const T f,std::vector<T>& vec){
    vec.push_back(f);
}

template<typename T1,typename T2>
void copy_elements(const std::vector<T1>& f,std::vector<T2>& vec){
    for(int i = 0;i<f.size();i++){
        copy_elements(f[i],vec);
    }
}

template <typename T>
class Tensor{
    public:
    std::vector<T> arr;
    std::vector<int> shape;
    int require_grad;
    template <typename T1>
    Tensor(std::vector<int> _shape,const std::vector<T1>& _vec,int _require_grad = 1){
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
        int size = 1;
        require_grad = _require_grad;
        for(int s : _shape){
            size *= s;
        }
        arr = std::vector<T>(size,0);
        shape = _shape; 
    }
};



template <typename T>
Tensor<T> operator+(const Tensor<T>& a,const Tensor<T>& b){
    if(a.shape != b.shape){
        throw 200;
    }
    Tensor<T> ret(a.shape);
    for(int i = 0;i<a.arr.size();i++){
        ret.arr[i] = a.arr[i] + b.arr[i];
    }
    return ret;
};

template <typename T>
Tensor<T> operator*(const float& a,const Tensor<T>& b){
    Tensor<T> ret(b.shape);
    for(int i = 0;i<b.arr.size();i++){
        ret.arr[i] = a * b.arr[i];
    }
    return ret;
};

template<typename T>
class Node{
    std::vector<Tensor<T>> operands;


};


int main(){
    try{
        Tensor<float> t(std::vector<int>({2,2}),std::vector<std::vector<float>>({{10,2},{3,4}}));
        Tensor<float> b(std::vector<int>({2,2}),std::vector<std::vector<float>>({{11,2},{3,4}}));
        Tensor<float> q = t + b;
        printvec((2*q).arr);
    }
    catch(int e){
        std::cout<<e<<std::endl;
    }
}