from argparse import ArgumentParser

import tensorrt as trt


parser = ArgumentParser(description="Create TensorRT Engine.")
parser.add_argument(
    "--batch_size", type=int, required=True,
    help="Batch size."
)
parser.add_argument(
    "--onnx", type=str, required=True,
    help="Path of onnx model."
)
parser.add_argument(
    "--out",  type=str, required=True,
    help="Output path of engine."
)
args = parser.parse_args()

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

EXPLICIT_BATCH = 1 << (int)(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
with trt.Builder(TRT_LOGGER) as builder:
    #builder.max_batch_size = args.batch_size
    builder.fp16_mode = True
    with builder.create_builder_config() as config:
        config.max_workspace_size = 1 << 30 # This determines the amount of memory available to the builder when building an optimized engine and should generally be set as high as possible.
        with builder.create_network(EXPLICIT_BATCH) as network, trt.OnnxParser(network, TRT_LOGGER) as parser:
            with open(args.onnx, 'rb') as model:
                print('Beginning ONNX file parsing')
                parser.parse(model.read())
            print('Completed parsing of ONNX file')
            print('Building an engine from file {}; this may take a while...'.format(args.onnx))
            with builder.build_engine(network, config) as engine:
                with open("model.trt", "wb") as f:
                    f.write(engine.serialize())
