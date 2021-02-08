import os
import os.path as osp
import random

import numpy as np
import torch
from loguru import logger


def init_train_env(
    work_dir: str, 
    seed: int
    ):
    """Initialize training environment.
    Args:
        work_dir (str): Directory to save log file, tensorboard file, checkpoint.
        seed (int): Random seed.
    """
    os.makedirs(work_dir, exist_ok=True)
    logger.add(osp.join(work_dir, "train.log"))
    torch.backends.cudnn.benchmark = True

    random.seed(seed)                                      
    torch.manual_seed(seed)                                
    torch.cuda.manual_seed(seed)                           
    np.random.seed(seed)


def get_map(
    query_loader: torch.utils.data.DataLoader, 
    gallery_loader: torch.utils.data.DataLoader,
    model: torch.nn.Module, 
    bits: int, 
    device: torch.device,
    topk: int,
    ) -> float:
    """Calculate mean average precision.

    Args:
        query_loader (torch.utils.data.DataLoader): Query data loader.
        gallery_loader (torch.utils.data.DataLoader): Gallery data loader.
        model (torch.nn.Module): CNN model.
        bits (int): Length of hash codes.
        device (torch.device): GPU device.
        topk (int): Calculate top-k mAP.

    Returns:
        float: mean average precision
    """
    model.eval()
    query_codes = generate_codes(query_loader, bits, model, device)
    gallery_codes = generate_codes(gallery_loader, bits, model, device)
    query_labels = query_loader.dataset.onehot_labels.to(device)
    gallery_labels = gallery_loader.dataset.onehot_labels.to(device)
    mAP = calculate_map(query_codes, gallery_codes, query_labels, gallery_labels, device, topk)
    model.train()

    return mAP

def generate_codes(
    loader: torch.utils.data.DataLoader,  
    bits: int, 
    model: torch.nn.Module, 
    device: torch.device, 
    ) -> torch.Tensor:
    """Generate hash codes.

    Args:
        loader (torch.utils.data.DataLoader): Data loader.
        bits (int): Length of hash codes.
        model (torch.nn.Module): CNN model.
        device (torch.device): GPU Device.

    Returns:
        torch.Tensor: Hash codes.
    """
    hash_codes = torch.zeros(len(loader.dataset), bits, device=device)
    with torch.no_grad():
        for x, _, index in loader:
            x = x.to(device)
            hash_code = model(x)
            hash_code = hash_code.sign()
            hash_codes[index, :] = hash_code

    return hash_codes 

def calculate_map(
    query_code: torch.Tensor,
    gallery_code: torch.Tensor,
    query_targets: torch.Tensor,
    gallery_targets: torch.Tensor,
    device: torch.device,
    topk: int=None,
) -> float:
    """
    Calculate mean average precision(map).
    Args:
        query_code (torch.Tensor): Query data hash code.
        gallery_code (torch.Tensor): Gallery data hash code.
        query_targets (torch.Tensor): Query data targets, one-hot
        gallery_targets (torch.Tensor): Gallery data targets, one-host
        device (torch.device): GPU Device.
        topk (int): Calculate top k data map.
    Returns:
        float: Mean Average Precision.
    """
    num_query = query_targets.shape[0]
    mean_AP = 0.0

    for i in range(num_query):
        # Retrieve images from database
        gallery = (query_targets[i, :] @ gallery_targets.t() > 0).float()

        # Calculate hamming distance
        hamming_dist = 0.5 * (gallery_code.shape[1] - query_code[i, :] @ gallery_code.t())

        # Arrange position according to hamming distance
        gallery = gallery[torch.argsort(hamming_dist)][:topk]

        # Retrieval count
        gallery_cnt = gallery.sum().int().item()

        # Can not retrieve images
        if gallery_cnt == 0:
            continue

        # Generate score for every position
        score = torch.linspace(1, gallery_cnt, gallery_cnt).to(device)

        # Acquire index
        index = (torch.nonzero(gallery == 1).squeeze() + 1.0).float()

        mean_AP += (score / index).mean()

    mean_AP = mean_AP / num_query

    return mean_AP.item()
