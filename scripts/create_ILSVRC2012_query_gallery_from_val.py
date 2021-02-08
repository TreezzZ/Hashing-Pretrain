import os
import os.path as osp
import random
import shutil

QUERY_PER_CLASS = 10
DATA_DIR = "/dataset/ILSVRC2012/"

os.makedirs(osp.join(DATA_DIR, "query"), exist_ok=True)
os.makedirs(osp.join(DATA_DIR, "gallery"), exist_ok=True)

for class_ in os.listdir(osp.join(DATA_DIR, "val")):
    val_class_dir = osp.join(DATA_DIR, "val", class_)
    query_class_dir = osp.join(DATA_DIR, "query", class_)
    gallery_class_dir = osp.join(DATA_DIR, "gallery", class_)
    os.makedirs(query_class_dir, exist_ok=True)
    os.makedirs(gallery_class_dir, exist_ok=True)

    files = os.listdir(val_class_dir)
    random.shuffle(files)
    query_files = files[:QUERY_PER_CLASS]
    gallery_files = files[QUERY_PER_CLASS:]

    [shutil.copy(osp.join(val_class_dir, file), osp.join(query_class_dir, file)) for file in query_files]
    [shutil.copy(osp.join(val_class_dir, file), osp.join(gallery_class_dir, file)) for file in gallery_files]
