import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
import numpy as np

class IrisNetwork(nn.Module):
    def __init__(self):
        super(IrisNetwork, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(4, 8),
            nn.ReLU(),
            nn.Linear(8, 8),
            nn.ReLU(),
            nn.Linear(8, 3),
            nn.Sigmoid()  # Add sigmoid for BCE loss
        )
    
    def forward(self, x):
        return self.layers(x)

def load_iris_pytorch():
    # Load iris dataset
    iris = load_iris()
    X = iris.data.astype(np.float32)
    y = iris.target
    
    # Convert to one-hot encoding
    y_onehot = np.zeros((y.size, 3), dtype=np.float32)
    y_onehot[np.arange(y.size), y] = 1
    
    # Convert to PyTorch tensors
    X_tensor = torch.from_numpy(X)
    y_tensor = torch.from_numpy(y_onehot)
    
    return X_tensor, y_tensor

def main():
    # Load data
    X_data, y_data = load_iris_pytorch()
    
    # Create model
    model = IrisNetwork()
    
    # BCE Loss function
    criterion = nn.BCELoss()
    
    # Training loop
    for epoch in range(60):
        total_loss = 0.0
        total_correct = 0.0
        
        for data_idx in range(len(X_data)):
            # Forward pass
            x_sample = X_data[data_idx].unsqueeze(0)  # Add batch dimension
            y_sample = y_data[data_idx].unsqueeze(0)  # Add batch dimension
            
            output = model(x_sample)
            
            # Calculate accuracy
            pred_idx = torch.argmax(output, dim=1).item()
            true_idx = torch.argmax(y_sample, dim=1).item()
            
            if pred_idx == true_idx:
                total_correct += 1
            
            # Calculate BCE loss
            loss = criterion(output, y_sample)
            total_loss += loss.item()
            
            # Backward pass
            loss.backward()
            
            # Manual parameter update (matching your learning rate)
            with torch.no_grad():
                for param in model.parameters():
                    if param.grad is not None:
                        param -= 0.003 * param.grad
            
            # Zero gradients
            model.zero_grad()
        
        print(f"epoch : {epoch}")
        print(f"total loss : {total_loss}")
        print(f"total correct : {total_correct}")

if __name__ == "__main__":
    main()
