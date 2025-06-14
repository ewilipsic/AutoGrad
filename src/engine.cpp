#include"node.h"
#include<stack>
#include<set>
#include<queue>

void backward(tensor start){

    // give out degrees

    std::stack<tensor> s;
    std::set<tensor> done;
    s.push(start);
    done.insert(start);

    while(!s.empty()){
        tensor current_tensor = s.top();s.pop();

        if((*current_tensor.grad_fn())){
            for(tensor child :(*current_tensor.grad_fn())->operands){
                child.out_degree()++;
                if(done.count(child)) continue;
                s.push(child);
                done.insert(child);
            }
        }
    }

    std::queue<tensor> q;
    q.push(start);
    start.grad()->fill(1.0);

    while (!q.empty())
    {
        tensor current_tensor = q.front();q.pop();
        current_tensor._backward();
        if((*current_tensor.grad_fn())){
            for(tensor child :(*current_tensor.grad_fn())->operands){
                child.out_degree()--;
                if(child.out_degree() == 0){
                    q.push(child);
                }
            }

            delete *(current_tensor.grad_fn());
            *(current_tensor.grad_fn()) = nullptr;
        }
    }
    





}