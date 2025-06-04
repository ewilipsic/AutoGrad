import torch

a = torch.tensor([1.2,2],requires_grad=True)
v = a[0] * a[1]
v.backward(gradient=torch.tensor(1.))
print(a.grad)