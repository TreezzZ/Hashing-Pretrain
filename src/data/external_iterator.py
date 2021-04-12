import collections
import copy
import os
import os.path as osp
import random

import numpy as np
import torch
from torchvision.datasets import ImageFolder
from tqdm import tqdm

class ExternalFolderTrainIterator(object):
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
        ):
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
        return self.n

    def _balanced_sample(self):
        indices = []
        class_per_batch = int(self.batch_size / self.sample_per_class)
        classes = np.random.choice(range(len(self.classes)), size=class_per_batch, replace=False)
        for class_idx in classes:
                all_indices_of_one_class = self.dataset[self.classes[class_idx]]
                for k in np.random.choice(range(len(all_indices_of_one_class)), size=self.sample_per_class, replace=False):
                    indices.append(all_indices_of_one_class[k])
        return indices


class ExternalFolderTestIterator(object):
    class_id_to = None

    def __init__(
        self,
        data_dir: str,
        batch_size: int,
    ):
        self.batch_size = batch_size
        if not ExternalFolderTestIterator.class_id_to:
            ExternalFolderTestIterator.class_to_id = {v: k for k, v in enumerate(os.listdir(data_dir))}
        self.img_pths = []
        self.labels = []
        for class_name in os.listdir(data_dir):
            class_id = ExternalFolderTestIterator.class_to_id[class_name]
            for file in os.listdir(osp.join(data_dir, class_name)):
                self.img_pths.append(osp.join(data_dir, class_name, file))
                self.labels.append(class_id)
        self.labels = torch.tensor(self.labels).long()
        self.onehot_labels = torch.zeros(len(self.labels), 1000) 
        for i in range(len(self.labels)):
            self.onehot_labels[i, self.labels[i]] = 1
        self.labels = self.onehot_labels
    
    def __len__(self):
        return len(self.img_pths)
    
    def __iter__(self):
        self.i = 0
        self.n = len(self.img_pths)
        return self

    def __next__(self):
        batch = []
        labels = []

        for _ in range(self.batch_size):
            img_filename = self.img_pths[self.i]
            label = self.labels[self.i]
            with open(img_filename, "rb") as f:
                batch.append(np.frombuffer(f.read(), dtype=np.uint8))
            labels.append(label)
            self.i += 1
            if self.i >= self.n:
                self.__iter__()
                raise StopIteration
        
        return (batch, labels)

    next = __next__


class ExternalTxtIterator(object):
    def __init__(
        self, 
        txt_pth: str, 
        batch_size: int,
        ):
        """External Text Iterator.

        Args:
            data_dir (str): Path of data txt file.
            batch_size (int): Batch size.
        """
        self.batch_size = batch_size

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
    
    def __len__(self):
        return len(self.img_pths)
    
    def __iter__(self):
        self.i = 0
        self.n = len(self.img_pths)
        return self

    def __next__(self):
        batch = []
        labels = []

        for _ in range(self.batch_size):
            img_filename = self.img_pths[self.i]
            label = self.labels[self.i]
            with open(img_filename, "rb") as f:
                batch.append(np.frombuffer(f.read(), dtype=np.uint8))
            labels.append(label)
            self.i += 1
            if self.i >= self.n:
                self.__iter__()
                raise StopIteration
        
        return (batch, labels)

    next = __next__
