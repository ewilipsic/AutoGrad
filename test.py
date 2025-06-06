import torch

a = torch.tensor([[1.2,2],[2,2]],requires_grad=True)
b = torch.tensor([[1.2,1.2],[22,2]],requires_grad=True)
v = a + b
v = v.sum()
print(v)
v.backward(gradient=torch.tensor(1.))
print(a.grad.requires_grad)