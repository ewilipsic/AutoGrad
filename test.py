import torch
import torch.nn as nn

# Create input tensor with shape (1, 1, 5, 5)
input_tensor = torch.zeros(1, 1, 5, 5)

# Fill input with simple pattern for easy verification (values 1-25)
input_tensor = input_tensor.view(-1)  # Flatten to fill
for i in range(25):
    input_tensor[i] = i + 1
input_tensor = input_tensor.view(1, 1, 5, 5)  # Reshape back

# Alternative way to fill the tensor (more Pythonic):
# input_tensor = torch.arange(1, 26, dtype=torch.float32).view(1, 1, 5, 5)

# Create Conv2D layer with specified parameters
# out_channels=1, in_channels=1, kernel_size=3x3, stride=1, padding=0
conv = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3, stride=1, padding=0, bias=True)

# Set the weight (edge detection kernel)
with torch.no_grad():
    conv.weight[0, 0] = torch.tensor([
        [-1, -1, -1],
        [-1,  4, -1],
        [-1, -1, -1]
    ], dtype=torch.float32)
    
    # Set bias
    conv.bias[0] = 0

# Forward pass
output = conv(input_tensor)

# Check output shape
assert output.shape[0] == 1  # batch
assert output.shape[1] == 1  # out_channels  
assert output.shape[2] == 3  # height: (5-3)/1+1 = 3
assert output.shape[3] == 3  # width: (5-3)/1+1 = 3

# Print output values (equivalent to printvec)
print("Output tensor:")
print(output.flatten().tolist())
print("\nOutput shape:", output.shape)

