import os.path as osp

import torch
import torchvision.transforms as T
from PIL import Image
from pytorch_metric_learning import samplers
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import ImageFolder


def get_dataloader(
    ilsvrc_data_dir,
    cifar_data_dir,
    nuswide_data_dir,
    train_batch_size,
    test_batch_size,
    samples_per_class,
    num_workers,
):
    # Get ILSVRC-2012 train data loader
    ilsvrc_train_dataloader = _get_train_dataloader(
        osp.join(ilsvrc_data_dir, "train"),
        train_batch_size,
        samples_per_class,
        num_workers,
    )

    # Get CIFAR-10 query and gallery data loader
    cifar_query_dataloader = _get_test_dataloader(
        osp.join(cifar_data_dir, "query.txt"),
        test_batch_size,
        num_workers,
    )
    cifar_gallery_dataloader = _get_test_dataloader(
        osp.join(cifar_data_dir, "gallery.txt"),
        test_batch_size,
        num_workers,
    )

    # Get NUS-WIDE query and gallery data loader
    nuswide_query_dataloader = _get_test_dataloader(
        osp.join(nuswide_data_dir, "query.txt"),
        test_batch_size,
        num_workers,
    )
    nuswide_gallery_dataloader = _get_test_dataloader(
        osp.join(nuswide_data_dir, "gallery.txt"),
        test_batch_size,
        num_workers,
    )

    return ilsvrc_train_dataloader, cifar_query_dataloader, cifar_gallery_dataloader, nuswide_query_dataloader, nuswide_gallery_dataloader


def _get_train_dataloader(
    data_dir,
    batch_size,
    samples_per_class,
    num_workers,
):
    dataset = ImageFolder(data_dir, transform=_get_transforms(is_train=True))
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        pin_memory=True,
        num_workers=num_workers,
        sampler=samplers.MPerClassSampler(dataset.targets, samples_per_class, batch_size)
    )
    return dataloader


def _get_test_dataloader(
    data_dir,
    batch_size,
    num_workers,
):
    dataset = TxtFormatDataset(data_dir, transform=_get_transforms(is_train=False))
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        pin_memory=True,
        num_workers=num_workers,
        shuffle=False,
    )
    return dataloader


def _get_transforms(is_train=True):
    transforms = []
    if is_train:
        transforms.extend([
            T.RandomResizedCrop(224),
            T.RandomHorizontalFlip(0.5),
        ])
    else:
        transforms.extend([
            T.Resize(256),
            T.CenterCrop(224),
        ]) 
    transforms.extend([
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225]),
    ])

    return T.Compose(transforms)


class TxtFormatDataset(Dataset):
    def __init__(self, txt_pth, transform=None):
        super(TxtFormatDataset, self).__init__()
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
