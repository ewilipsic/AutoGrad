import torch

# Create tensors with requires_grad=True
a = torch.full((3, 3), 2.0, requires_grad=True)
b = torch.full((3, 3), 1.0, requires_grad=True)

# Compute the operations
c = torch.square(a)                    # c = a^2
d = c + b                             # d = c + b = a^2 + b
e = torch.square(d) + torch.sqrt(b/9)  # e = d^2 + 9*sqrt(b)

# Retain gradients for intermediate tensors
c.retain_grad()
d.retain_grad()
e.retain_grad()

# Backward pass
e_sum = e.sum()
e_sum.backward()

# Print gradients
print("a.grad:\n", a.grad)
print("b.grad:\n", b.grad)
print("c.grad:\n", c.grad)
print("d.grad:\n", d.grad)
print("e.grad:\n", e.grad)
