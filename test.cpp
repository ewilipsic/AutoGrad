#include<bits/stdc++.h>

using namespace std;

class E{
    public:
    int n;
    E* one;
    E(int _n){n = _n;}

    ~E(){
        cout<<"Destroyed";
    }
};
int main(){
    E e(1);

    
}