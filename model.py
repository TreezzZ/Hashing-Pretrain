import torchvision
import torch.nn as nn


def get_model(
    arch: str, 
    num_dim: int,
    ) -> nn.Module:
    """Create CNN model.

    Args:
        arch (str): Model name.
        num_dim (int): Number of output dimension.

    Returns:
        torch.nn.Module: CNN model.
    """
    if arch.lower() == "googlenet":
        model = torchvision.models.googlenet(pretrained=True)
        model.fc = nn.Linear(1024, num_dim)
    else:
        raise ValueError("Can not find CNN model name!")

    return model
