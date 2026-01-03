#!/usr/bin/env python3
"""
Export MNIST model weights to binary format for Mini-Inference Engine.

This script trains a simple MLP on MNIST and exports the weights
in the format expected by the inference engine.
"""

import struct
import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torchvision import datasets, transforms
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("PyTorch not available. Using random weights.")


# Weight file format constants
WEIGHT_FILE_MAGIC = 0x4D494E49  # "MINI"
WEIGHT_FILE_VERSION = 1


class MNISTNet(nn.Module):
    """Simple MLP for MNIST classification."""
    
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 10)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = x.view(-1, 784)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def train_mnist_model(epochs=5):
    """Train MNIST model and return it."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on {device}")
    
    # Data loading
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST('./data', train=False, transform=transform)
    
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1000)
    
    # Model
    model = MNISTNet().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    
    # Training
    for epoch in range(epochs):
        model.train()
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            if batch_idx % 200 == 0:
                print(f'Epoch {epoch+1}/{epochs} [{batch_idx * len(data)}/{len(train_loader.dataset)}] '
                      f'Loss: {loss.item():.4f}')
        
        # Test
        model.eval()
        correct = 0
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                pred = output.argmax(dim=1)
                correct += pred.eq(target).sum().item()
        
        accuracy = 100. * correct / len(test_loader.dataset)
        print(f'Epoch {epoch+1} Test Accuracy: {accuracy:.2f}%')
    
    return model


def export_weights(model, output_path):
    """Export model weights to binary format."""
    layers = [
        (model.fc1.weight.data.cpu().numpy(), 
         model.fc1.bias.data.cpu().numpy() if model.fc1.bias is not None else None),
        (model.fc2.weight.data.cpu().numpy(),
         model.fc2.bias.data.cpu().numpy() if model.fc2.bias is not None else None),
        (model.fc3.weight.data.cpu().numpy(),
         model.fc3.bias.data.cpu().numpy() if model.fc3.bias is not None else None),
    ]
    
    with open(output_path, 'wb') as f:
        # Write header
        f.write(struct.pack('I', WEIGHT_FILE_MAGIC))
        f.write(struct.pack('I', WEIGHT_FILE_VERSION))
        f.write(struct.pack('I', len(layers)))
        f.write(b'\x00' * 20)  # Reserved
        
        # Write each layer
        for weight, bias in layers:
            # PyTorch stores weights as (out_features, in_features)
            # We need (in_features, out_features) for row-major C
            weight = weight.T.astype(np.float32)
            
            in_features, out_features = weight.shape
            has_bias = 1 if bias is not None else 0
            
            # Layer meta
            f.write(struct.pack('I', 0))  # type = Linear
            f.write(struct.pack('I', in_features))
            f.write(struct.pack('I', out_features))
            f.write(struct.pack('I', has_bias))
            
            # Weights (row-major)
            f.write(weight.tobytes())
            
            # Bias
            if bias is not None:
                f.write(bias.astype(np.float32).tobytes())
    
    print(f"Weights exported to {output_path}")


def create_random_weights(output_path):
    """Create random weights for testing without PyTorch."""
    np.random.seed(42)
    
    layers = [
        (784, 256),
        (256, 128),
        (128, 10)
    ]
    
    with open(output_path, 'wb') as f:
        # Write header
        f.write(struct.pack('I', WEIGHT_FILE_MAGIC))
        f.write(struct.pack('I', WEIGHT_FILE_VERSION))
        f.write(struct.pack('I', len(layers)))
        f.write(b'\x00' * 20)  # Reserved
        
        for in_features, out_features in layers:
            # Xavier initialization
            scale = np.sqrt(2.0 / (in_features + out_features))
            weight = np.random.randn(in_features, out_features).astype(np.float32) * scale
            bias = np.zeros(out_features, dtype=np.float32)
            
            # Layer meta
            f.write(struct.pack('I', 0))  # type = Linear
            f.write(struct.pack('I', in_features))
            f.write(struct.pack('I', out_features))
            f.write(struct.pack('I', 1))  # has_bias
            
            # Weights
            f.write(weight.tobytes())
            
            # Bias
            f.write(bias.tobytes())
    
    print(f"Random weights created at {output_path}")


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Export MNIST weights')
    parser.add_argument('--output', '-o', default='mnist_weights.bin',
                        help='Output file path')
    parser.add_argument('--epochs', '-e', type=int, default=5,
                        help='Training epochs')
    parser.add_argument('--random', '-r', action='store_true',
                        help='Use random weights instead of training')
    args = parser.parse_args()
    
    if args.random or not TORCH_AVAILABLE:
        create_random_weights(args.output)
    else:
        model = train_mnist_model(args.epochs)
        export_weights(model, args.output)


if __name__ == '__main__':
    main()
