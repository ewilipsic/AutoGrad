import torch

# Create tensors with the same values and enable gradient computation
a = torch.tensor([[-1., -1., 2.], 
                  [2., -3., 4.], 
                  [2., -3., 4.]], requires_grad=True)

b = torch.tensor([[-1., 1., 2.], 
                  [2., 3., 4.], 
                  [9., 1., 4.]], requires_grad=True)

# Perform the same operations: c = a + b + matmul(a,b)
c = a + b + torch.matmul(a, b)

# Compute gradients (backward pass)
# Since c is a matrix, we need to provide a gradient tensor of the same shape
# Using ones() assumes we want the gradient with respect to the sum of all elements
c.backward(torch.ones_like(c))

# Print the gradients
print("Gradient of a:")
print(a.grad)
print("\nGradient of b:")
print(b.grad)
