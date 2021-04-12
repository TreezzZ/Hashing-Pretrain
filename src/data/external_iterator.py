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
        samples_per_class: int,
        length_before_new_iter: int=1000000,
        ):
        """External Input Iterator.

        Args:
            data_dir (str): Dictory of dataset.
            batch_size (int): Batch size.
            samples_per_class (int): Number of samples per class.
            length_before_new_iter (int): Iteration length.
        """
        self.batch_size = batch_size
        self.samples_per_class = samples_per_class
        self.length_before_new_iter = length_before_new_iter

        dataset = ImageFolder(data_dir)
        # ((img1, label1), (img2, label2), ...)
        self.samples = dataset.samples

        self.data = [item[0] for item in self.samples]
        self.labels = [item[1] for item in self.samples]
        self.labels_to_indices = self._get_labels_to_indices(self.labels)

    def _get_labels_to_indices(self, labels):
        """
        Creates labels_to_indices, which is a dictionary mapping each label
        to a numpy array of indices that will be used to index into self.dataset
        """
        if torch.is_tensor(labels):
            labels = labels.cpu().numpy()
        labels_to_indices = collections.defaultdict(list)
        for i, label in enumerate(labels):
            labels_to_indices[label].append(i)
        for k, v in labels_to_indices.items():
            labels_to_indices[k] = np.array(v, dtype=np.int)
        return labels_to_indices
    
    def __iter__(self):
        idx_list = [0] * self.length_before_new_iter
        i = 0
        self.num_iters = self.length_before_new_iter // self.batch_size
        labels = copy.deepcopy(self.labels)
        for _ in tqdm(range(self.num_iters), desc="Prepare data"):
            random.shuffle(labels)
            curr_label_set = labels[: self.batch_size // self.samples_per_class]
            for label in curr_label_set:
                t = self.labels_to_indices[label]
                idx_list[i : i + self.samples_per_class] = self._safe_random_choice(
                    t, size=self.samples_per_class
                )
                i += self.samples_per_class
        self.idx_list = idx_list
        self.i = 0
        return self


    def _safe_random_choice(self, input_data, size):
        """
        Randomly samples without replacement from a sequence. It is "safe" because
        if len(input_data) < size, it will randomly sample WITH replacement
        Args:
            input_data is a sequence, like a torch tensor, numpy array,
                            python list, tuple etc
            size is the number of elements to randomly sample from input_data
        Returns:
            An array of size "size", randomly sampled from input_data
        """
        replace = len(input_data) < size
        return np.random.choice(input_data, size=size, replace=replace)

    def __next__(self):
        idx = self.idx_list[self.i : min(self.i + self.batch_size, self.length_before_new_iter)]
        self.i += self.batch_size
        if self.i > self.length_before_new_iter:
            self.__iter__()
            raise StopIteration

        imgs = []
        labels = []
        for i in idx:
            img_filename = self.data[i]
            label = self.labels[i]
            with open(img_filename, 'rb') as f:
                imgs.append(np.frombuffer(f.read(), dtype=np.uint8))
            labels.append(np.array([label], dtype=np.uint8))
        
        return (imgs, labels)

    next = __next__

    def __len__(self):
        return self.num_iters


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
