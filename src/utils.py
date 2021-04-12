import os
import os.path as osp
import random

import numpy as np
import torch
from loguru import logger


def init_train_env(
    work_dir: str, 
    seed: int,
    gpu: int,
    ):
    """Initialize training environment.
    Args:
        work_dir (str): Directory to save log file, tensorboard file, checkpoint.
        seed (int): Random seed.
        gpu (int): GPU ID.
    """
    os.makedirs(work_dir, exist_ok=True)
    logger.add(osp.join(work_dir, "train.log"))
    torch.backends.cudnn.benchmark = True
    torch.cuda.set_device(gpu)

    random.seed(seed)                                      
    torch.manual_seed(seed)                                
    torch.cuda.manual_seed(seed)                           
    np.random.seed(seed)


def get_map(
    query_loader: torch.utils.data.DataLoader, 
    gallery_loader: torch.utils.data.DataLoader,
    model: torch.nn.Module, 
    bits: int, 
    topk: int,
    ) -> float:
    """Calculate mean average precision.

    Args:
        query_loader (torch.utils.data.DataLoader): Query data loader.
        gallery_loader (torch.utils.data.DataLoader): Gallery data loader.
        model (torch.nn.Module): CNN model.
        bits (int): Length of hash codes.
        topk (int): Calculate top-k mAP.

    Returns:
        float: mean average precision
    """
    model.eval()
    query_codes, query_labels = generate_codes(query_loader, bits, model)
    gallery_codes, gallery_labels = generate_codes(gallery_loader, bits, model)
    mAP = calculate_map(query_codes, gallery_codes, query_labels, gallery_labels, topk)
    model.train()

    return mAP

def generate_codes(
    loader: torch.utils.data.DataLoader,  
    bits: int, 
    model: torch.nn.Module, 
    ) -> torch.Tensor:
    """Generate hash codes.

    Args:
        loader (torch.utils.data.DataLoader): Data loader.
        bits (int): Length of hash codes.
        model (torch.nn.Module): CNN model.

    Returns:
        torch.Tensor: Hash codes.
    """
    hash_codes = torch.zeros(len(loader._pipes[0].input_iter), bits).cuda()
    labels = loader._pipes[0].input_iter.labels.cuda()
    pointer = 0
    with torch.no_grad():
        for batch in loader:
            x = batch[0]["data"]
            hash_code = model(x)
            hash_code = hash_code.sign()

            next_pointer = pointer + x.shape[0]
            hash_codes[pointer: next_pointer, :] = hash_code
            pointer = next_pointer
        loader.reset()

    return hash_codes, labels

def calculate_map(
    query_code: torch.Tensor,
    gallery_code: torch.Tensor,
    query_targets: torch.Tensor,
    gallery_targets: torch.Tensor,
    topk: int=None,
) -> float:
    """
    Calculate mean average precision(map).
    Args:
        query_code (torch.Tensor): Query data hash code.
        gallery_code (torch.Tensor): Gallery data hash code.
        query_targets (torch.Tensor): Query data targets, one-hot
        gallery_targets (torch.Tensor): Gallery data targets, one-host
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
        score = torch.linspace(1, gallery_cnt, gallery_cnt).cuda()

        # Acquire index
        index = (torch.nonzero(gallery == 1).squeeze() + 1.0).float()

        mean_AP += (score / index).mean()

    mean_AP = mean_AP / num_query

    return mean_AP.item()
