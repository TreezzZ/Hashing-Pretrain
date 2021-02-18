import tensorrt as trt
from argparse import ArgumentParser

parser = ArgumentParser(description="TensorRT")
parser.add_argument(
    "--checkpoint",  type=str, required=True,
    help="Path of onnx model."
)
args = parser.parse_args()

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

# Create builder, network and parser
EXPLICIT_BATCH = 1 << (int)(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
with trt.Builder(TRT_LOGGER) as builder, builder.create_network(EXPLICIT_BATCH) as network, trt.OnnxParser(network, TRT_LOGGER) as parser:
    with open(args.checkpoint, 'rb') as model:
        if not parser.parse(model.read()):
            for error in range(parser.num_errors):
                print(parser.get_error(error))

# Create engine
with trt.Builder(TRT_LOGGER) as builder, builder.create_builder_config() as config:
    config.max_workspace_size = 1 << 20 # This determines the amount of memory available to the builder when building an optimized engine and should generally be set as high as possible.
    with builder.build_engine(network, config) as engine:
        # Save engine
        with open("hash_pretrain.engine", "wb") as f:
            f.write(engine.serialize())
