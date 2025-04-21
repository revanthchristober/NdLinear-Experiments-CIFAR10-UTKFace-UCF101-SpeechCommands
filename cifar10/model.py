# ndlinear_project/cifar10/model.py
import torch.nn as nn
import torch.nn.functional as F
from ndlinear import NdLinear # Assuming ndlinear is installed

class NdLinearCIFAR10(nn.Module):
    """
    A CNN model for CIFAR-10 that integrates NdLinear.
    Expects input shape [B, 3, 32, 32].
    """
    def __init__(self, ndlinear_hidden_size=(4, 4, 128)):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(2, 2) # Output: [B, 64, 8, 8]

        # NdLinear expects input_dims matching (H, W, C) after permute
        self.ndlinear = NdLinear((8, 8, 64), ndlinear_hidden_size) # Output: [B, H', W', C'] e.g., [B, 4, 4, 128]

        # Calculate flattened size from ndlinear_hidden_size
        flat_size = 1
        for dim in ndlinear_hidden_size:
            flat_size *= dim # e.g., 4 * 4 * 128 = 2048

        self.fc = nn.Linear(flat_size, 10) # 10 classes for CIFAR-10

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool2(x) # Shape: [B, 64, 8, 8]

        # Permute for NdLinear: [B, H, W, C]
        x = x.permute(0, 2, 3, 1).contiguous() # Shape: [B, 8, 8, 64]

        x = self.ndlinear(x) # Shape: [B, H', W', C']

        # Flatten and classify
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x