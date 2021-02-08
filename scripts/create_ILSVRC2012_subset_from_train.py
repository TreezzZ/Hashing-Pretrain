import os
import os.path as osp
import shutil
import random

SAMPLE_PER_CLASS = 100
DATA_DIR = "/dataset/ILSVRC2012/train"
OUT_DIR = "/dataset/ILSVRC2012_subset"

os.makedirs(OUT_DIR, exist_ok=True)
OUT_DIR = osp.join(OUT_DIR, "train")
os.makedirs(OUT_DIR, exist_ok=True)

for class_ in os.listdir(DATA_DIR):
    class_dir = osp.join(DATA_DIR, class_)
    os.makedirs(osp.join(OUT_DIR, class_), exist_ok=True)
    files = os.listdir(class_dir)
    random.shuffle(files)
    for file in files[:SAMPLE_PER_CLASS]:
        shutil.copy(osp.join(class_dir, file), osp.join(OUT_DIR, class_, file))
