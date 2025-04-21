# ndlinear_project/utkface/model.py
import torch
import torch.nn as nn
import torchvision
from ndlinear import NdLinear

class NdLinearUTKFace(nn.Module):
    """
    NdLinear model for UTKFace dataset (Age, Gender, Race prediction).
    Uses a ResNet18 backbone. Expects input [B, 3, 128, 128].
    """
    def __init__(self, ndlinear_hidden_size=(2, 2, 128)):
        super().__init__()
        # 1) Load pre-trained ResNet18 and remove the final avgpool and fc layers
        resnet = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1)
        self.backbone = nn.Sequential(*list(resnet.children())[:-2])
        # Output of backbone for [B, 3, 128, 128] input is [B, 512, 4, 4]

        # 2) NdLinear layer expects input_dims=(H, W, C) after permute
        # Input to NdLinear will be (4, 4, 512)
        self.nd = NdLinear((4, 4, 512), ndlinear_hidden_size)
        # Output shape from NdLinear: [B, H', W', C'] e.g., [B, 2, 2, 128]

        # 3) Calculate flattened size from ndlinear_hidden_size for prediction heads
        flat_size = 1
        for dim in ndlinear_hidden_size:
            flat_size *= dim # e.g., 2 * 2 * 128 = 512

        # 4) Prediction heads
        self.age_head    = nn.Linear(flat_size, 1)  # Regression
        self.gender_head = nn.Linear(flat_size, 2)  # Binary classification (0 Male, 1 Female)
        self.race_head   = nn.Linear(flat_size, 5)  # Multi-class classification (0 White, 1 Black, 2 Asian, 3 Indian, 4 Others)

    def forward(self, x):
        # x: [B, 3, 128, 128]
        x = self.backbone(x)
        # x: [B, 512, 4, 4]

        # Permute for NdLinear: [B, H, W, C]
        x = x.permute(0, 2, 3, 1).contiguous()
        # x: [B, 4, 4, 512]

        x = self.nd(x)
        # x: [B, H', W', C'] e.g., [B, 2, 2, 128]

        # Flatten for prediction heads
        x = x.view(x.size(0), -1)
        # x: [B, flat_size] e.g., [B, 512]

        # Get predictions from heads
        age_pred = self.age_head(x)
        gender_pred = self.gender_head(x)
        race_pred = self.race_head(x)

        return age_pred, gender_pred, race_pred