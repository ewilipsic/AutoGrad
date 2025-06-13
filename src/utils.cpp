#include "tensor.h"
#include <iostream>

void printvec(std::vector<float> vec) {
    for(auto& x : vec) std::cout << x << " ";
    std::cout << std::endl;
}

void printvec(std::vector<int> vec) {
    for(auto& x : vec) std::cout << x << " ";
    std::cout << std::endl;
}

int get_element_count(const float f) {
    return 1;
}

template<typename T>
int get_element_count(const std::vector<T>& vec) {
    int ret = 0;
    for(int i = 0; i < vec.size(); i++) {
        ret += get_element_count(vec[i]);
    }
    return ret;
}

void copy_elements(const float f, std::vector<float>& vec) {
    vec.push_back(f);
}

template<typename T1>
void copy_elements(const std::vector<T1>& f, std::vector<float>& vec) {
    for(int i = 0; i < f.size(); i++) {
        copy_elements(f[i], vec);
    }
}

// Explicit template instantiations
template int get_element_count(const std::vector<float>& vec);
template int get_element_count(const std::vector<std::vector<float>>& vec);
template void copy_elements(const std::vector<float>& f, std::vector<float>& vec);
template void copy_elements(const std::vector<std::vector<float>>& f, std::vector<float>& vec);
