#include<iostream>
using namespace std;

class Enitity{
    public:
    virtual void func() = 0;
};

class Person : public Enitity{
    public:
    void func(){
        cout<<10<<endl;
    }
};
class Person2 : public Enitity{
    public:
    void func(){
        cout<<12<<endl;
    }
};
int main(){
    Enitity* t = new Person2;

    t->func();
}