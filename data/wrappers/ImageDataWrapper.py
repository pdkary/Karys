from typing import List
from data.configs.ImageDataConfig import ImageDataConfig
from data.loaders import ImageDataLoader


class ImageDataWrapper(object):
    def __init__(self, image_set: List, data_ref: ImageDataConfig):
        self.image_set = image_set
        self.data_ref = data_ref
        self.train_dataset = None
        self.test_dataset = None

    @classmethod
    def load_from_directory(cls, dirpath, image_type, data_ref: ImageDataConfig, load_n_percent=100):
        image_set = ImageDataLoader.load_images(dirpath, image_type, load_n_percent)
        return cls(image_set, data_ref)
    
    def load_datasets(self):
        train_length = int(len(self.image_set)*self.data_ref.train_test_ratio)
        train_imgs = self.image_set[0:train_length]
        test_imgs = self.image_set[train_length:]
        self.train_dataset = ImageDataLoader.load_dataset(train_imgs, self.data_ref)
        self.test_dataset = ImageDataLoader.load_dataset(test_imgs, self.data_ref)
    
    def get_train_dataset(self):
        if self.train_dataset is None:
            self.load_datasets()
        return self.train_dataset

    def get_test_dataset(self):
        if self.test_dataset is None:
            self.load_datasets()
        return self.test_dataset

    def __len__(self):
        return len(self.image_set)
