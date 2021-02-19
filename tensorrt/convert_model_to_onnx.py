import sys
sys.path.append('..')

import torch
import onnx
from src.model import get_model
from argparse import ArgumentParser

parser = ArgumentParser(description="Convert PyTorch Model to ONNX Format")
parser.add_argument(
    "--arch", type=str, required=True,
    help="CNN model name."
)
parser.add_argument(
    "--bits", type=int, required=True,
    help="Hash code length."
)
parser.add_argument(
    "--batch_size", type=int, required=True,
    help="Batch size."
)
parser.add_argument(
    "--checkpoint", type=str, required=True,
    help="Path of Checkpoint."
)
parser.add_argument(
    "--out",  type=str, required=True,
    help="Output path of onnx model."
)
parser.add_argument(
    "--gpu", type=int, default=0,
    help="Specify gpu. (Default: 0)"
)
args = parser.parse_args()
args.device = torch.device("cuda", args.gpu)

# Load model
print("Loading model...")
model = get_model(args.arch, args.bits)
model.load_state_dict(torch.load(args.checkpoint))
model.eval()
model.to(args.device)
print("finished.")

# Create random tensor with output shape
x = torch.randn(args.batch_size, 3, 224, 224, requires_grad=True, device=args.device)
outputs = model(x)

# Export ONNX model
print("Exporting model...")
torch.onnx.export(
    model,
    x,
    args.out,
    export_params=True,
    opset_version=10,
    do_constant_folding=True,
    input_names=["input"],
    output_names=["output"],
    #dynamic_axes={'input' : {0 : 'batch_size'},    # variable lenght axes
    #              'output' : {0 : 'batch_size'}},
)
print("finished.")

# Check ONNX model
print("Checking model...")
onnx_model = onnx.load(args.out)
onnx.checker.check_model(onnx_model)
print("finished.")