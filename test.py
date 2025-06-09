import torch

# Create tensors with the same values and enable gradient computation
a = torch.tensor([[-1.0, -1.0, 2.0], 
                  [2.0, -3.0, 4.0], 
                  [2.0, -3.0, 4.0]], requires_grad=True)

b = torch.tensor([[-1.0, 1.0, 2.0], 
                  [2.0, 3.0, 4.0], 
                  [9.0, 1.0, 4.0]], requires_grad=True)

# Forward pass - retain gradients for intermediate tensors
c = torch.relu(a)
c.retain_grad()

d = torch.matmul(b, c)
d.retain_grad()

# Backward pass
loss = d.sum()
loss.backward()

# Print only gradients (flattened)
print(a.grad.flatten().tolist())
print(b.grad.flatten().tolist())
print(c.grad.flatten().tolist())
print(d.grad.flatten().tolist())
