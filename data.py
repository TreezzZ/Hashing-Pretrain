import os
import os.path as osp
from typing import Dict, List, Tuple

import numpy as np
import nvidia.dali.fn as fn
import nvidia.dali.ops as ops
import nvidia.dali.types as types
import torch
import torchvision.transforms as T
from nvidia.dali.pipeline import Pipeline
from nvidia.dali.plugin.pytorch import DALIGenericIterator, LastBatchPolicy
from PIL import Image
from torch.utils.data import DataLoader, Dataset


def get_train_dataloader(
    ilsvrc_data_dir: str,
    batch_size: int,
    sample_per_class: int,
    num_workers: int,
    gpu_id: int,
    ) -> torch.utils.data.DataLoader:
    """Create train data loader.

    Args:
        ilsvrc_data_dir (str): Dictory of ILSVRC-2012.
        batch_size (int): Batch size.
        sample_per_class (int): Number of samples per class.
        num_workers (int): Number of data loader threads.
        gpu_id (int): GPU id.

    Returns:
        torch.utils.data.DataLoader: ILSVRC-2012 train data loader.
    """
    ilsvrc_data_dir = osp.join(ilsvrc_data_dir, "train")
    ilsvrc_train_pipeline = ILSVRC_train_Pipeline(
        ilsvrc_data_dir,
        batch_size=batch_size, 
        sample_per_class=sample_per_class,
        device_id=gpu_id, 
        num_threads=num_workers,
    )
    ilsvrc_train_pipeline.build()
    ilsvrc_train_loader = DALIGenericIterator(
        [ilsvrc_train_pipeline],
        ["data", "label"],
        last_batch_padded=True,
        last_batch_policy=LastBatchPolicy.PARTIAL,
    )

    return ilsvrc_train_loader


class ExternalInputIterator(object):
    def __init__(
        self, 
        data_dir: str, 
        batch_size: int,
        sample_per_class: int,
        ):
        """External Input Iterator.

        Args:
            data_dir (str): Dictory of dataset.
            batch_size (int): Batch size.
            sample_per_class (int): Number of samples per class.
        """
        self.batch_size = batch_size
        self.sample_per_class = sample_per_class
        self.data = []
        self.labels = []

        classes, class_to_idx = self._find_classes(data_dir)
        self.classes = classes
        self.dataset = {class_: [] for class_ in classes}

        idx = 0
        for class_ in classes:
            cur_class = osp.join(data_dir, class_)
            files = os.listdir(cur_class)
            files = [os.path.join(cur_class, i) for i in files]
            self.data.extend(files)
            self.labels.extend([class_to_idx[class_] for i in range(len(files))])
            self.dataset[class_].extend([idx+i for i in range(len(files))])
            idx += len(files)
        
        num_samples = [len(value) for value in self.dataset.values()]
        self.max_samples = max(num_samples)
        self.min_samples = min(num_samples)

        assert self.min_samples >= self.sample_per_class
    
    def _find_classes(
        self, 
        data_dir: str,
        ) -> Tuple[List[str], Dict[str, int]]:
        """
        Finds the class folders in a dataset.
        Args:
            data_dir (string): Root directory path.
        Returns:
            tuple: (classes, class_to_idx) where classes are relative to (dir), and class_to_idx is a dictionary.
        Ensures:
            No class is a subdirectory of another.
        """
        classes = [d.name for d in os.scandir(data_dir) if d.is_dir()]
        classes.sort()
        class_to_idx = {classes[i]: i for i in range(len(classes))}
        return classes, class_to_idx

    def __iter__(self):
        self.i = 0
        self.n = len(self.data) // self.batch_size
        return self

    def __next__(self):
        if self.i >= self.n:
            self.i = 0
            raise StopIteration
        self.i += 1

        batch = []
        labels = []
        indices = self._balanced_sample()
        
        for idx in indices:
            img_filename = self.data[idx]
            label = self.labels[idx]
            with open(img_filename, 'rb') as f:
                batch.append(np.frombuffer(f.read(), dtype = np.uint8))
            labels.append(np.array([label], dtype = np.uint8))
        
        return (batch, labels)

    next = __next__

    def __len__(self):
        return len(self.data)

    def _balanced_sample(self) -> List[int]:
        indices = []
        class_per_batch = int(self.batch_size / self.sample_per_class)
        classes = np.random.choice(range(len(self.classes)), size=class_per_batch, replace=False)
        for class_idx in classes:
                all_indices_of_one_class = self.dataset[self.classes[class_idx]]
                for k in np.random.choice(range(len(all_indices_of_one_class)), size=self.sample_per_class, replace=False):
                    indices.append(all_indices_of_one_class[k])
        return indices


class ILSVRC_train_Pipeline(Pipeline):
    def __init__(
        self, 
        data_dir: str, 
        batch_size: int, 
        sample_per_class: int,
        device_id: int = 0,
        num_threads: int = 4, 
        shard_id: int = 0,
        num_shards: int = 1, 
        seed: int = 1996,
        ):
        """ILSVRC-2012 training pipeline.

        Args:
            data_dir (str): Dictory of dataset.
            batch_size (int): Batch size.
            sample_per_class (int): Samples per class.
            device_id (int, optional): GPU id. Defaults to 0.
            num_threads (int, optional): Number of data loader threads. Defaults to 4.
            shard_id (int, optional): Shared id. Defaults to 0.
            num_shards (int, optional): Number of shared. Defaults to 1.
            seed (int, optional): Random seed. Defaults to 1996.
        """
        super(ILSVRC_train_Pipeline, self).__init__(batch_size, num_threads, device_id, seed)
        self.input_iter = iter(ExternalInputIterator(data_dir, batch_size, sample_per_class))
        self.input_imgs = ops.ExternalSource()
        self.input_labels = ops.ExternalSource()

        self.decode = ops.ImageDecoder(device="mixed", output_type=types.RGB)
        self.res = ops.Resize(device="gpu", resize_shorter=256, interp_type=types.INTERP_TRIANGULAR)
        self.cmn = ops.CropMirrorNormalize(
            device="gpu",
            output_layout=types.NCHW,
            crop=(224, 224),
            mean=[0.485 * 255, 0.456 * 255, 0.406 * 255],
            std=[0.229 * 255, 0.224 * 255, 0.225 * 255])
        self.coin = ops.CoinFlip(probability=0.5)
        self.to_int64 = ops.Cast(dtype=types.INT64, device="gpu")
    
    def iter_setup(self):
        (images, labels) = self.input_iter.next()
        self.feed_input(self.imgs, images)
        self.feed_input(self.labels, labels)

    def define_graph(self):
        self.imgs = self.input_imgs()
        self.labels = self.input_labels()

        images = self.decode(self.imgs)
        images = self.res(images)
        images = self.cmn(images.gpu(), mirror=self.coin())
        labels = self.labels.gpu()
        labels = self.to_int64(labels)

        return (images, labels)


def get_test_dataloader(
    ilsvrc_data_dir: str, 
    cifar_data_dir: str, 
    nuswide_data_dir: str, 
    batch_size: int, 
    num_workers: int,
    ) -> Tuple[torch.utils.data.DataLoader]:
    """Create test data loader.

    Args:
        ilsvrc_data_dir (str): Dictory of ILSVRC-2012.
        cifar_data_dir (str): Dictory of CIFAR-10.
        nuswide_data_dir (str): Dictory of NUS-WIDE.
        batch_size (int): Batch size.
        num_workers (int): Number of data loader threads.

    Returns:
        Tuple[torch.utils.data.DataLoader]: ILSVRC-2012, CIFAR-10, NUS-WIDE data loaders.
    """
    test_transform = T.Compose([
        T.Resize(256),
        T.CenterCrop(224),
        T.ToTensor(),
        #T.ConvertImageDtype(torch.float32),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    ilsvrc_query_dataset = ImageNet(
        osp.join(ilsvrc_data_dir, "query"),
        test_transform,
    )
    ilsvrc_query_loader = DataLoader(
        ilsvrc_query_dataset,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=True,
        num_workers=num_workers   
    )
    ilsvrc_gallery_dataset = ImageNet(
        osp.join(ilsvrc_data_dir, "gallery"),
        test_transform,
    )
    ilsvrc_gallery_loader = DataLoader(
        ilsvrc_gallery_dataset,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=True,
        num_workers=num_workers   
    )

    cifar_query_dataset = HashingDataset(osp.join(cifar_data_dir, "query.txt"), test_transform)
    cifar_query_loader = DataLoader(
        cifar_query_dataset,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=True,
        num_workers=num_workers
    )
    cifar_gallery_dataset = HashingDataset(osp.join(cifar_data_dir, "gallery.txt"), test_transform)
    cifar_gallery_loader = DataLoader(
        cifar_gallery_dataset,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=True,
        num_workers=num_workers
    )

    nuswide_query_dataset = HashingDataset(osp.join(nuswide_data_dir, "query.txt"), test_transform)
    nuswide_query_loader = DataLoader(
        nuswide_query_dataset,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=True,
        num_workers=num_workers
    )
    nuswide_gallery_dataset = HashingDataset(osp.join(nuswide_data_dir, "gallery.txt"), test_transform)
    nuswide_gallery_loader = DataLoader(
        nuswide_gallery_dataset,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=True,
        num_workers=num_workers
    )

    return ilsvrc_query_loader, ilsvrc_gallery_loader, cifar_query_loader, cifar_gallery_loader, nuswide_query_loader, nuswide_gallery_loader


class ImageNet(Dataset):
    class_id_to = None
    def __init__(self, data_dir, transform=None):
        super(ImageNet, self).__init__()
        self.transform = transform
        if not ImageNet.class_id_to:
            ImageNet.class_to_id = {v: k for k, v in enumerate(os.listdir(data_dir))}
        self.img_pths = []
        self.labels = []
        for class_name in os.listdir(data_dir):
            class_id = ImageNet.class_to_id[class_name]
            for file in os.listdir(osp.join(data_dir, class_name)):
                self.img_pths.append(osp.join(data_dir, class_name, file))
                self.labels.append(class_id)
        self.labels = torch.tensor(self.labels).long()
        self.onehot_labels = torch.zeros(len(self.labels), 1000) 
        for i in range(len(self.labels)):
            self.onehot_labels[i, self.labels[i]] = 1
    
    def __getitem__(self, index):
        img_pth = self.img_pths[index]
        label = self.labels[index]
        #img = tio.read_image(img_pth)
        img = Image.open(img_pth).convert("RGB")
        if self.transform:
            img = self.transform(img)

        return img, label, index

    def __len__(self):
        return len(self.img_pths)


class HashingDataset(Dataset):
    def __init__(self, txt_pth, transform=None):
        super(HashingDataset, self).__init__()
        self.transform = transform

        img_dir = osp.join("/".join(txt_pth.split("/")[:-1]), "images")
        self.img_pths = []
        self.labels = []
        with open(txt_pth, "r") as f:
            for line in f:
                line = line.strip().split(" ")
                img_pth = line[0]
                label = list(map(int, line[1:]))
                self.img_pths.append(osp.join(img_dir, img_pth))
                self.labels.append(torch.tensor(label))
        self.labels = torch.vstack(self.labels).float()
        self.onehot_labels = self.labels
        
    def __getitem__(self, index):
        img_pth = self.img_pths[index]
        label = self.labels[index]
        img = Image.open(img_pth).convert("RGB")
        if self.transform:
            img = self.transform(img)
        
        return img, label, index

    def __len__(self):
        return len(self.img_pths)


if __name__ == "__main__":
    dataloader = get_train_dataloader(
        "/dataset/ILSVRC2012",
        128,
        8,
        6,
        0,
    )

    import time
    start = time.time()
    for d in dataloader:
        img = d[0]["data"]
        labels = d[0]["label"]
    print("time: {:.4f}".format(time.time() - start))
