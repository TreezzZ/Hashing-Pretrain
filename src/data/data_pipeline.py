import nvidia.dali.fn as fn
import nvidia.dali.ops as ops
import nvidia.dali.types as types
from nvidia.dali.pipeline import Pipeline

from .external_iterator import (ExternalFolderTestIterator,
                                ExternalFolderTrainIterator,
                                ExternalTxtIterator)


class TrainPipeline(Pipeline):
    def __init__(
        self, 
        data_dir: str, 
        batch_size: int, 
        samples_per_class: int,
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
            samples_per_class (int): Number of samples per class.
            device_id (int, optional): GPU id. Defaults to 0.
            num_threads (int, optional): Number of data loader threads. Defaults to 4.
            shard_id (int, optional): Shared id. Defaults to 0.
            num_shards (int, optional): Number of shared. Defaults to 1.
            seed (int, optional): Random seed. Defaults to 1996.
        """
        super(TrainPipeline, self).__init__(batch_size, num_threads, device_id, seed)
        self.input_iter = iter(ExternalFolderTrainIterator(data_dir, batch_size, samples_per_class))
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


class TestPipeline(Pipeline):
    def __init__(
        self, 
        data_dir: str, 
        folder: bool,
        batch_size: int, 
        device_id: int = 0,
        num_threads: int = 4, 
        shard_id: int = 0,
        num_shards: int = 1, 
        seed: int = 1996,
        ):
        """Test pipeline.
        Args:
            data_dir (str): Dictory of dataset.
            folder (bool): True if folder else txt.
            batch_size (int): Batch size.
            device_id (int, optional): GPU id. Defaults to 0.
            num_threads (int, optional): Number of data loader threads. Defaults to 4.
            shard_id (int, optional): Shared id. Defaults to 0.
            num_shards (int, optional): Number of shared. Defaults to 1.
            seed (int, optional): Random seed. Defaults to 1996.
        """
        super(TestPipeline, self).__init__(batch_size, num_threads, device_id, seed)
        if folder:
            self.input_iter = iter(ExternalFolderTestIterator(data_dir, batch_size))
        else:
            self.input_iter = iter(ExternalTxtIterator(data_dir, batch_size))
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


def get_train_pipeline(
    data_dir: str,
    batch_size: int,
    samples_per_class: int,
    num_workers: int,
    device_id: int,
    seed: int,
):
    """Get train pipeline.

    Args:
        data_dir (str): Directory of train dataset.
        batch_size (int): Batch size,
        samples_per_class (int): Number of samples per class.
        num_workers (int): umber of data loader threads.
        device_id (int): GPU id.
        seed (int): Seed.

    Returns:
        Train data loader.
    """
    pipeline = Pipeline(batch_size, num_workers, device_id, seed)
    with pipeline:
        eii = ExternalFolderTrainIterator(
            data_dir,
            batch_size,
            samples_per_class,
        )
        images, labels = fn.external_source(source=eii, num_outputs=2)
        images = fn.image_decoder(
            images,
            device="mixed",
            output_type=types.RGB,
        ).gpu()
        images = fn.random_resized_crop(
            images,
            size=(224, 224),
        )
        #images = fn.coin_flip(images)
        images = fn.normalize(
            images,
            mean=[0.485 * 255, 0.456 * 255, 0.406 * 255],
            stddev=[0.229 * 255, 0.224 * 255, 0.225 * 255],
        )
        labels = labels.gpu()
        pipeline.set_outputs(images, labels)
    return pipeline


def get_test_pipeline(
    data_dir: str,
    batch_size: int,
    num_workers: int,
    device_id: int,
    folder: bool,
    seed: int,
):
    pipeline = Pipeline(batch_size, num_workers, device_id, seed)
    with pipeline:
        if folder:
            eii = ExternalFolderTestIterator(data_dir, batch_size)
        else:
            eii = ExternalTxtIterator(data_dir, batch_size)
        images, labels = fn.external_source(source=eii, num_outputs=2)
        images = fn.image_decoder(
            images,
            device="mixed",
            output_type=types.RGB,
        ).gpu()
        images = fn.resize(
            images,
            device="gpu",
            size=256,
            mode="not_smaller",
            interp_type=types.INTERP_TRIANGULAR,
        )
        images = fn.crop_mirror_normalize(
            images,
            device="gpu",
            dtype=types.FLOAT,
            output_layout="CHW",
            crop=(224, 224),
            mean=[0.485 * 255,0.456 * 255,0.406 * 255],
            std=[0.229 * 255,0.224 * 255,0.225 * 255],
            mirror=False,
        )
        labels = labels.gpu()
        pipeline.set_outputs(images, labels)
    return pipeline
