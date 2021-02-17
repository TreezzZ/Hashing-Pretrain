import datetime
import os.path as osp
from argparse import ArgumentParser

import torch


def get_config():
    """Return configurations.
    Returns:
        ArgumentParser: Configurations.
    """
    parser = ArgumentParser(description="PretrainedHashing")
    hash_group = parser.add_argument_group("Hashing")
    hash_group.add_argument(
        "--arch", type=str, default="googlenet",
        help="CNN model name. (Default: GoogLeNet)"
    )
    hash_group.add_argument(
        "--bits", type=int, required=True,
        help="Hash code length."
    )
    hash_group.add_argument(
        "--margin", type=float, default=1.0,
        help="Margin. (Default: 1.0)"
    )

    dataset_group = parser.add_argument_group("Dataset")
    dataset_group.add_argument(
        "--ilsvrc_data_dir", type=str, required=True,
        help="Directory of ILSVRC-2012."
    )
    dataset_group.add_argument(
        "--cifar_data_dir", type=str, required=True,
        help="Directory of CIFAR-10."
    )
    dataset_group.add_argument(
        "--nuswide_data_dir", type=str, required=True,
        help="Directory of NUS-WIDE."
    )
    dataset_group.add_argument(
        "--batch_size", type=int, default=128,
        help="Batch size. (default: 128)"
    )
    dataset_group.add_argument(
        "--sample_per_class", type=int, default=8,
        help="Number of samples per class. (default: 8)"
    )
    dataset_group.add_argument(
        "--num_workers", type=int, default=6,
        help="The number of thread workers to load data. (default: 6)"
    )

    optimizer_group = parser.add_argument_group("Optimizer")
    optimizer_group.add_argument(
        "--lr", type=float, default=1e-2,
        help="Learning rate. (default: 1e-2)"
    )
    optimizer_group.add_argument(
        "--weight_decay", type=float, default=1e-5,
        help="Weight decay. (default: 1e-5)"
    )

    train_group = parser.add_argument_group("Train")
    train_group.add_argument(
        "--gpu", type=int, default=0,
        help="Specify gpu. (Default: 0)"
    )
    train_group.add_argument(
        "--max_epochs", type=int, default=5,
        help="The number of epochs. (default: 5)"
    )
    train_group.add_argument(
        "--eval_step", type=int, default=5,
        help="Evaluate validation dataset every n epochs. (default: 5)"
    )
    train_group.add_argument(
        "--work_dir", type=str, default="./work_dir",
        help="The directory stores log files, checkpoints, visualization files and so on. (default: work_dir)"
    )
    train_group.add_argument(
        "--seed", type=int, default=24,
        help="Random Seed. (default: 24)"
    )
    train_group.add_argument(
        "--fp16", action="store_true",
        help="FP16 Training."
    )
    train_group.add_argument(
        "--tensorboard", action="store_true",
        help="Enable tensorboard."
    )
    train_group.add_argument(
        "--no_save", action="store_true",
        help="Do not save checkpoint."
    )
    train_group.add_argument(
        "--log_step", type=int, default=50,
        help="Log information every n steps (default: 50)."
    )

    args = parser.parse_args()
    args.device = torch.device("cuda", args.gpu)
    args.work_dir = osp.join(args.work_dir, f"{str(datetime.datetime.now())}")

    return args
