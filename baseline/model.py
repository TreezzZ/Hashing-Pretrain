import torchvision
import torch.nn as nn
import copy


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
        hash_layer = nn.Sequential(
            nn.Linear(1024, num_dim),
            nn.Tanh(),
        )
        classification_layer = nn.Linear(num_dim, 1000)
        model.fc = nn.Sequential(
            hash_layer,
            classification_layer
        )
    elif arch.lower() == "alexnet":
        model = torchvision.models.alexnet(pretrained=True)
        hash_layer = nn.Sequential(
            nn.Linear(4096, num_dim),
            nn.Tanh(),
        )
        classification_layer = nn.Linear(num_dim, 1000)
        model.classifier = nn.Sequential(
            model.classifier[:-1],
            hash_layer,
            classification_layer,
        )
    elif arch.lower() == "mobilenet":
        model = torchvision.models.mobilenet_v2(pretrained=True)
        hash_layer = nn.Sequential(
            nn.Linear(model.last_channel, num_dim),
            nn.Tanh(),
        )
        classification_layer = nn.Linear(num_dim, 1000)
        model.classifier = nn.Sequential(
            model.classifier[0],
            hash_layer,
            classification_layer,
        )
    else:
        raise ValueError("Can not find CNN model name!")

    return model
