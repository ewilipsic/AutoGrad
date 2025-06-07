import torch

# Create tensors with requires_grad=True
a = torch.tensor([[3.0], [2.0], [1.0]], requires_grad=True)
b = torch.tensor([[3.0], [2.0], [1.0]], requires_grad=True)
c = torch.tensor([[2.0], [1.0], [1.0]], requires_grad=True)
d = torch.tensor([[1.0], [2.0], [1.0]], requires_grad=True)

# Perform operations
e = b + c
f = a + c
g = e + f

# Retain gradients for intermediate tensors
e.retain_grad()
f.retain_grad()
g.retain_grad()

print("AAA")
# Backward pass
g.backward(torch.ones_like(g))

# Print all gradients
print("a.grad:", a.grad)
print("b.grad:", b.grad)
print("c.grad:", c.grad)
print("d.grad:", d.grad)
print("e.grad:", e.grad)
print("f.grad:", f.grad)
print("g.grad:", g.grad)