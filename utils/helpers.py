# ndlinear_project/utils/helpers.py
import torch
import torch.nn as nn

def count_parameters(model: nn.Module) -> int:
    """Counts the number of trainable parameters in a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def get_device(no_cuda: bool = False) -> torch.device:
    """Gets the appropriate torch device (CUDA or CPU)."""
    use_cuda = not no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    print(f"Using device: {device}")
    return device