import sys
sys.path.append('..')
import time

from argparse import ArgumentParser

import numpy as np
import torch
from src.data import get_test_dataloader, get_dali_test_dataloader
from src.utils import calculate_map
from src.model import get_model
from inference import *
from PIL import Image
import torchvision.transforms as T

parser = ArgumentParser(description="TensorRT Inference")
parser.add_argument(
    "--arch", type=str, required=True,
    help="CNN model name."
)
parser.add_argument(
    "--engine_pth", type=str, required=True,
    help="Path of TensorRT engine."
)
parser.add_argument(
    "--model_pth", type=str, required=True,
    help="Path of PyTorch model."
)
parser.add_argument(
    "--batch_size", type=int, required=True,
    help="Batch Size."
)
parser.add_argument(
    "--bits", type=int, required=True,
    help="The length of hash codes."
)
parser.add_argument(
    "--data_type", type=str, required=True,
    help="Data type."
)
parser.add_argument(
    "--ilsvrc_data_dir", type=str, required=True,
    help="Directory of ILSVRC-2012."
)
parser.add_argument(
    "--cifar_data_dir", type=str, required=True,
    help="Directory of CIFAR-10."
)
parser.add_argument(
    "--nuswide_data_dir", type=str, required=True,
    help="Directory of NUS-WIDE."
)
parser.add_argument(
    "--num_workers", type=int, default=6,
    help="The number of thread workers to load data. (default: 6)"
)
parser.add_argument(
    "--gpu", type=int, default=0,
    help="Specify gpu. (Default: 0)"
)

args = parser.parse_args()
args.device = torch.device("cuda", args.gpu)
if args.data_type == "fp32":
    args.data_type = np.float32
elif args.data_type == "fp16":
    args.data_type = np.float16
elif args.data_type == "int8":
    args.data_type = np.int8
else:
    raise ValueError(f"Can not find data type: {args.data_type}, please check it!")


def dali_generate_codes(
    loader: torch.utils.data.DataLoader,  
    bits: int, 
    context,
    stream,
    bindings, 
    inputs, 
    outputs
    ) -> torch.Tensor:
    """Generate hash codes.

    Args:
        loader (torch.utils.data.DataLoader): Data loader.
        bits (int): Length of hash codes.
        model (torch.nn.Module): CNN model.

    Returns:
        torch.Tensor: Hash codes.
    """
    hash_codes = np.zeros((loader.size, bits))
    labels = None
    idx = 0
    for batch in loader:
        data = batch[0]["data"]
        label = batch[0]["label"].squeeze()

        if labels is None:
            labels = torch.zeros((loader.size, label.shape[1]))
        labels[idx:idx+len(label), :] = label.cpu()

        np.copyto(inputs[0].host, data.cpu().numpy().ravel().astype(np.float32))
        [results] = do_inference_v2(context, bindings, inputs, outputs, stream)
        hash_codes[idx:idx+len(label), :] = np.sign(results.reshape(-1, bits))
        idx += len(label)
    return hash_codes, labels


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
        

if __name__ == "__main__":
    # TensorRT
    # Prepare data
    dali_ilsvrc_query_loader, dali_ilsvrc_gallery_loader, dali_cifar_query_loader, dali_cifar_gallery_loader, dali_nuswide_query_loader, dali_nuswide_gallery_loader = get_dali_test_dataloader(args.ilsvrc_data_dir, args.cifar_data_dir, args.nuswide_data_dir, args.batch_size, args.num_workers)

    # Load engine
    with open(args.engine_pth, "rb") as f:
        runtime = trt.Runtime(trt.Logger(trt.Logger.WARNING)) 
        engine = runtime.deserialize_cuda_engine(f.read())

    # Allocate memory
    inputs, outputs, bindings, stream = allocate_buffers(engine)
    with engine.create_execution_context() as context:
        # Generate codes
        dali_ilsvrc_query_codes, dali_ilsvrc_query_labels = dali_generate_codes(dali_ilsvrc_query_loader, args.bits, context, stream, bindings, inputs, outputs)
        dali_ilsvrc_time = time.time()
        dali_ilsvrc_gallery_codes, dali_ilsvrc_gallery_labels = dali_generate_codes(dali_ilsvrc_gallery_loader, args.bits, context, stream, bindings, inputs, outputs)
        dali_ilsvrc_time = time.time() - dali_ilsvrc_time
        dali_ilsvrc_query_codes = torch.from_numpy(dali_ilsvrc_query_codes).to(args.device)
        dali_ilsvrc_gallery_codes = torch.from_numpy(dali_ilsvrc_gallery_codes).to(args.device)
        dali_ilsvrc_query_labels = dali_ilsvrc_query_labels.to(args.device)
        dali_ilsvrc_gallery_labels = dali_ilsvrc_gallery_labels.to(args.device)
        dali_ilsvrc_mAP = calculate_map(dali_ilsvrc_query_codes, dali_ilsvrc_gallery_codes, dali_ilsvrc_query_labels, dali_ilsvrc_gallery_labels, args.device, 1000)
        print("DALI ILSVRC-2012, mAP: {:.4f}, time: {:.2f}".format(dali_ilsvrc_mAP, dali_ilsvrc_time))

        dali_cifar_query_codes, dali_cifar_query_labels = dali_generate_codes(dali_cifar_query_loader, args.bits, context, stream, bindings, inputs, outputs)
        dali_cifar_time = time.time()
        dali_cifar_gallery_codes, dali_cifar_gallery_labels = dali_generate_codes(dali_cifar_gallery_loader, args.bits, context, stream, bindings, inputs, outputs)
        dali_cifar_time = time.time() - dali_cifar_time
        dali_cifar_query_codes = torch.from_numpy(dali_cifar_query_codes).to(args.device)
        dali_cifar_gallery_codes = torch.from_numpy(dali_cifar_gallery_codes).to(args.device)
        dali_cifar_query_labels = dali_cifar_query_labels.to(args.device)
        dali_cifar_gallery_labels = dali_cifar_gallery_labels.to(args.device)
        dali_cifar_mAP = calculate_map(dali_cifar_query_codes, dali_cifar_gallery_codes, dali_cifar_query_labels, dali_cifar_gallery_labels, args.device, -1)
        print("DALI CIFAR-10, mAP: {:.4f}, time: {:.2f}".format(dali_cifar_mAP, dali_cifar_time))

        dali_nuswide_query_codes, dali_nuswide_query_labels = dali_generate_codes(dali_nuswide_query_loader, args.bits, context, stream, bindings, inputs, outputs)
        dali_nuswide_time = time.time()
        dali_nuswide_gallery_codes, dali_nuswide_gallery_labels = dali_generate_codes(dali_nuswide_gallery_loader, args.bits, context, stream, bindings, inputs, outputs)
        dali_nuswide_time = time.time() - dali_nuswide_time
        dali_nuswide_query_codes = torch.from_numpy(dali_nuswide_query_codes).to(args.device)
        dali_nuswide_gallery_codes = torch.from_numpy(dali_nuswide_gallery_codes).to(args.device)
        dali_nuswide_query_labels = dali_nuswide_query_labels.to(args.device)
        dali_nuswide_gallery_labels = dali_nuswide_gallery_labels.to(args.device)
        dali_nuswide_mAP = calculate_map(dali_nuswide_query_codes, dali_nuswide_gallery_codes, dali_nuswide_query_labels, dali_nuswide_gallery_labels, args.device, 1000)
        print("DALI NUS-WIDE, mAP: {:.4f}, time: {:.2f}".format(dali_nuswide_mAP, dali_nuswide_time))


    # Non-TensorRT
    # Create model and load weights
    model = get_model(args.arch, args.bits).to(args.device)
    model.load_state_dict(torch.load(args.model_pth))
    model.eval()

    ilsvrc_query_loader, ilsvrc_gallery_loader, cifar_query_loader, cifar_gallery_loader, nuswide_query_loader, nuswide_gallery_loader = get_test_dataloader(args.ilsvrc_data_dir, args.cifar_data_dir, args.nuswide_data_dir, args.batch_size, args.num_workers)

    ilsvrc_query_codes = generate_codes(ilsvrc_query_loader, args.bits, model, args.device)
    ilsvrc_time = time.time()
    ilsvrc_gallery_codes = generate_codes(ilsvrc_gallery_loader, args.bits, model, args.device)
    ilsvrc_time = time.time() - ilsvrc_time
    ilsvrc_query_labels = ilsvrc_query_loader.dataset.onehot_labels.to(args.device)
    ilsvrc_gallery_labels = ilsvrc_gallery_loader.dataset.onehot_labels.to(args.device)
    ilsvrc_mAP = calculate_map(ilsvrc_query_codes, ilsvrc_gallery_codes, ilsvrc_query_labels, ilsvrc_gallery_labels, args.device, 1000)
    print("ILSVRC-2012, mAP: {:.4f}, time: {:.2f}".format(ilsvrc_mAP, ilsvrc_time))

    cifar_query_codes = generate_codes(cifar_query_loader, args.bits, model, args.device)
    cifar_time = time.time()
    cifar_gallery_codes = generate_codes(cifar_gallery_loader, args.bits, model, args.device)
    cifar_time = time.time() - cifar_time
    cifar_query_labels = cifar_query_loader.dataset.onehot_labels.to(args.device)
    cifar_gallery_labels = cifar_gallery_loader.dataset.onehot_labels.to(args.device)
    cifar_mAP = calculate_map(cifar_query_codes, cifar_gallery_codes, cifar_query_labels, cifar_gallery_labels, args.device, -1)
    print("CIFAR-10, mAP: {:.4f}, time: {:.2f}".format(cifar_mAP, cifar_time))

    nuswide_query_codes = generate_codes(nuswide_query_loader, args.bits, model, args.device)
    nuswide_time = time.time()
    nuswide_gallery_codes = generate_codes(nuswide_gallery_loader, args.bits, model, args.device)
    nuswide_time = time.time() - nuswide_time
    nuswide_query_labels = nuswide_query_loader.dataset.onehot_labels.to(args.device)
    nuswide_gallery_labels = nuswide_gallery_loader.dataset.onehot_labels.to(args.device)
    nuswide_mAP = calculate_map(nuswide_query_codes, nuswide_gallery_codes, nuswide_query_labels, nuswide_gallery_labels, args.device, 5000)
    print("NUS-WIDE, mAP: {:.4f}, time: {:.2f}".format(nuswide_mAP, nuswide_time))
