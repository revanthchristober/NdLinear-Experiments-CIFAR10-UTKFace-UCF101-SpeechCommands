# ndlinear_project/ucf101/model.py
import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights
from ndlinear import NdLinear

class NdLinearUCF101(nn.Module):
    """
    NdLinear model for UCF101 action recognition.
    Uses a ResNet18 backbone applied per frame.
    Expects input clip shape [B, T, C, H, W], e.g., [B, 4, 3, 128, 128].
    """
    def __init__(self, num_classes: int, frames_per_clip: int = 4, ndlinear_hidden_size=(2, 2, 2, 256)):
        super().__init__()
        self.frames_per_clip = frames_per_clip

        # 1) Pretrained ResNet18 trunk (remove avgpool and fc)
        resnet = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        self.backbone = nn.Sequential(*list(resnet.children())[:-2])
        # Backbone output for [B*T, 3, 128, 128] is [B*T, 512, 4, 4]

        # 2) NdLinear layer expects input_dims=(T, H, W, C) after view/permute
        # Input to NdLinear will be (frames_per_clip, 4, 4, 512)
        # Example: (4, 4, 4, 512)
        backbone_output_channels = 512 # From ResNet18 layer4
        backbone_output_spatial = 4    # Spatial size after backbone for 128x128 input
        nd_input_dims = (frames_per_clip, backbone_output_spatial, backbone_output_spatial, backbone_output_channels)

        # Ensure ndlinear_hidden_size has 4 dimensions (T', H', W', C')
        if len(ndlinear_hidden_size) != 4:
             raise ValueError(f"ndlinear_hidden_size must have 4 dimensions (T', H', W', C'), got {ndlinear_hidden_size}")

        self.nd = NdLinear(nd_input_dims, ndlinear_hidden_size)
        # Output shape from NdLinear: [B, T', H', W', C'] e.g., [B, 2, 2, 2, 256]

        # 3) Calculate flattened size from ndlinear_hidden_size for classifier
        flat_size = 1
        for dim in ndlinear_hidden_size:
            flat_size *= dim # e.g., 2 * 2 * 2 * 256 = 2048

        # 4) Final classifier
        self.fc = nn.Linear(flat_size, num_classes)

    def forward(self, clip):
        # clip: [B, T, C, H, W]
        B, T, C, H, W = clip.shape
        if T != self.frames_per_clip:
            # This shouldn't happen if data loading is correct, but good check
            raise ValueError(f"Input clip time dimension ({T}) doesn't match model's frames_per_clip ({self.frames_per_clip})")

        # Reshape to process frames independently through backbone: (B*T, C, H, W)
        x = clip.view(B * T, C, H, W)

        # Pass through backbone
        features_flat = self.backbone(x)
        # features_flat: [B*T, 512, 4, 4]

        # Reshape back to [B, T, C_out, H_out, W_out]
        features = features_flat.view(B, T, features_flat.shape[1], features_flat.shape[2], features_flat.shape[3])
        # features: [B, T=4, 512, 4, 4]

        # Permute for NdLinear: [B, T, H, W, C]
        features_permuted = features.permute(0, 1, 3, 4, 2).contiguous()
        # features_permuted: [B, 4, 4, 4, 512]

        # Pass through NdLinear
        nd_out = self.nd(features_permuted)
        # nd_out: [B, T', H', W', C'] e.g., [B, 2, 2, 2, 256]

        # Flatten for classifier
        out_flat = nd_out.view(B, -1)
        # out_flat: [B, flat_size] e.g., [B, 2048]

        # Classify
        logits = self.fc(out_flat)
        return logits