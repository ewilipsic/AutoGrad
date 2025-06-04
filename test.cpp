#include<bits/stdc++.h>
#include<memory>
using namespace std;

class E{
    public:
    int n;
    E(int _n){n = _n;}

    ~E(){
        cout<<"Destroyed";
    }
};
int main(){
    {
    shared_ptr<E> s;
    {
        E e(1);
        s = make_shared<E>(&e);
    }
    cout<<"AAAA";
    }

    
}