import os
import os.path as osp

from nvidia.dali.plugin.pytorch import DALIGenericIterator

from .data_pipeline import get_test_pipeline, get_train_pipeline, TrainPipeline, TestPipeline


def get_dataloader(
    ilsvrc_data_dir: str, 
    cifar_data_dir: str, 
    nuswide_data_dir: str, 
    batch_size: int, 
    samples_per_class: int,
    num_workers: int,
    device_id: int,
    seed: int,
):
    """Get dataloader.

    Args:
        ilsvrc_data_dir (str): Directory of the ILSVRC-2012 dataset.
        cifar_data_dir (str): Directory of the CIFAR-10 dataset.
        nuswide_data_dir (str): Directory of the NUS-WIDE dataset.
        batch_size (int): Batch size.
        samples_per_class (int): Number of samples per class.
        num_workers (int): Number of data loader threads.
        device_id (int): GPU ID.
        seed (int): Seed.
    
    Returns:
        Data loaders.
    """
    ilsvrc_train_loader = get_train_dataloader(
        osp.join(ilsvrc_data_dir, "train"),
        batch_size=batch_size,
        samples_per_class=samples_per_class,
        num_workers=num_workers,
        device_id=device_id,
        seed=seed,
    )
    ilsvrc_query_loader = get_test_dataloader(
        osp.join(ilsvrc_data_dir, "query"), 
        batch_size=batch_size,
        num_workers=num_workers,
        folder=True,
        device_id=device_id,
        seed=seed,
    )
    ilsvrc_gallery_loader = get_test_dataloader(
        osp.join(ilsvrc_data_dir, "gallery"), 
        batch_size=batch_size, 
        num_workers=num_workers, 
        folder=True,
        device_id=device_id,
        seed=seed,
    )

    cifar_query_loader = get_test_dataloader(
        osp.join(cifar_data_dir, "query.txt"), 
        batch_size=batch_size,
        num_workers=num_workers,
        folder=False,
        device_id=device_id,
        seed=seed,
    )
    cifar_gallery_loader = get_test_dataloader(
        osp.join(cifar_data_dir, "gallery.txt"), 
        batch_size=batch_size, 
        num_workers=num_workers, 
        folder=False,
        device_id=device_id,
        seed=seed,
    )

    nuswide_query_loader = get_test_dataloader(
        osp.join(nuswide_data_dir, "query.txt"), 
        batch_size=batch_size,
        num_workers=num_workers,
        folder=False,
        device_id=device_id,
        seed=seed,
    )
    nuswide_gallery_loader = get_test_dataloader(
        osp.join(nuswide_data_dir, "gallery.txt"), 
        batch_size=batch_size, 
        num_workers=num_workers, 
        folder=False,
        device_id=device_id,
        seed=seed,
    )

    return ilsvrc_train_loader, ilsvrc_query_loader, ilsvrc_gallery_loader, cifar_query_loader, cifar_gallery_loader, nuswide_query_loader, nuswide_gallery_loader


def get_train_dataloader(
    ilsvrc_data_dir: str,
    batch_size: int,
    samples_per_class: int,
    num_workers: int,
    device_id: int,
    seed: int,
    ):
    """Create train data loader.

    Args:
        ilsvrc_data_dir (str): Dictory of ILSVRC-2012.
        batch_size (int): Batch size.
        samples_per_class (int): Number of samples per class.
        num_workers (int): Number of data loader threads.
        device_id (int): GPU id.
        seed (int): Seed.

    Returns:
        torch.utils.data.DataLoader: Train data loader.
        int: Number of iterations.
    """
    #ilsvrc_train_pipeline = get_train_pipeline(
    #    ilsvrc_data_dir,
    #    batch_size,
    #    samples_per_class,
    #    num_workers,
    #    device_id,
    #    seed,
    #)
    ilsvrc_train_pipeline = TrainPipeline(
        ilsvrc_data_dir,
        batch_size,
        samples_per_class,
        device_id=device_id,
        num_threads=num_workers,
        seed=seed,
    )
    ilsvrc_train_pipeline.build()
    ilsvrc_train_loader = DALIGenericIterator(
        [ilsvrc_train_pipeline],
        ["data", "label"],
    )

    return ilsvrc_train_loader


def get_test_dataloader(
    data_dir: str,
    batch_size: int,
    num_workers: int,
    folder: bool,
    device_id: int,
    seed: int,
):
    """Get test data loader.

    Args:
        data_dir (str): Dictory of test dataset.
        batch_size (int): Batch size.
        samples_per_class (int): Number of samples per class.
        num_workers (int): Number of data loader threads.
        device_id (int): GPU id.
        seed (int): Seed.

    Returns:
        torch.utils.data.DataLoader: Test data loader.
    """
    #pipeline, size = get_test_pipeline(
    #    data_dir,
    #    batch_size=batch_size,
    #    num_workers=num_workers,
    #    device_id=device_id,
    #    folder=True,
    #    seed=seed,
    #)
    pipeline = TestPipeline(
        data_dir,
        folder,
        batch_size,
        device_id=device_id,
        num_threads=num_workers,
        seed=seed,
    )
    pipeline.build()
    loader = DALIGenericIterator(
        [pipeline],
        ["data", "label"],
    )

    return loader
