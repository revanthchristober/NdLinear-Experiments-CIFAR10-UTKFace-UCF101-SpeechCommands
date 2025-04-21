# ndlinear_project/speech_commands/model.py
import torch
import torch.nn as nn
from ndlinear import NdLinear

class NdLinearAudio(nn.Module):
    """
    NdLinear model for Google Speech Commands dataset classification.
    Uses a simple Conv2D backbone on Mel Spectrograms.
    Expects input [B, 1, F, T], e.g., [B, 1, 64, T_mel].
    """
    def __init__(self, num_classes: int, input_freq_bins: int = 64, ndlinear_hidden_size=(8, 8, 16)):
        super().__init__()
        # 1) Conv backbone: adaptively pool to ensure fixed size before NdLinear
        self.conv = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1), # Output F/2, T/2
            nn.ReLU(),
            nn.BatchNorm2d(16), # Added BatchNorm
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1), # Output F/4, T/4
            nn.ReLU(),
            nn.BatchNorm2d(32), # Added BatchNorm
            nn.AdaptiveAvgPool2d((16, 16)) # Force output spatial size to 16x16 -> [B, 32, 16, 16]
        )

        # 2) NdLinear expects input_dims=(Freq, Time, Channels) after permute
        # Input to NdLinear will be (16, 16, 32)
        nd_input_dims = (16, 16, 32) # F_pool, T_pool, C_conv

        # Ensure hidden size has 3 dimensions (F', T', C')
        if len(ndlinear_hidden_size) != 3:
             raise ValueError(f"ndlinear_hidden_size must have 3 dimensions (F', T', C'), got {ndlinear_hidden_size}")

        self.nd = NdLinear(input_dims=nd_input_dims, hidden_size=ndlinear_hidden_size)
        # Output shape: [B, F', T', C'] e.g., [B, 8, 8, 16]

        # 3) Calculate flattened size for classifier
        flat_size = 1
        for dim in ndlinear_hidden_size:
            flat_size *= dim # e.g., 8 * 8 * 16 = 1024

        # 4) Final classifier
        self.fc = nn.Linear(flat_size, num_classes)

    def forward(self, x):
        # x: [B, 1, F, T_mel] (Mel Spectrogram)
        B = x.size(0)

        # Pass through conv backbone
        x = self.conv(x) # Output: [B, 32, 16, 16]

        # Reorder for NdLinear: [B, F_pool, T_pool, C_conv]
        x = x.permute(0, 2, 3, 1).contiguous() # Output: [B, 16, 16, 32]

        # Pass through NdLinear
        x = self.nd(x) # Output: [B, F', T', C'] e.g., [B, 8, 8, 16]

        # Flatten
        x = x.view(B, -1) # Output: [B, flat_size] e.g., [B, 1024]

        # Classify
        logits = self.fc(x)
        return logits