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
    elif arch.lower() == "alexnet":
        model = torchvision.models.alexnet(pretrained=True)
        model.classifier[-1] = nn.Linear(4096, num_dim)
    elif arch.lower() == "mobilenet":
        model = torchvision.models.mobilenet_v2(pretrained=True)
        model.classifier[-1] = nn.Linear(model.last_channel, num_dim)
    else:
        raise ValueError("Can not find CNN model name!")

    return model
