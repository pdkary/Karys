

from typing import List
from data.configs.ImageDataConfig import ImageDataConfig
from data.loaders import ImageDataLoader

from data.wrappers.DataWrapper import DataWrapper


class ImageDataWrapper(DataWrapper):
    def __init__(self,image_set: List, data_config: ImageDataConfig, train_test_ratio: float = 0.7):
        super(ImageDataWrapper, self).__init__(train_test_ratio)
        self.image_set = image_set
        self.data_config = data_config
    
    @classmethod
    def load_from_file(cls,filename, data_config: ImageDataConfig, train_test_ratio: float = 0.7):
        images = ImageDataLoader.load_images(filename, data_config)
        return cls(images, data_config, train_test_ratio)
    
    def get_train_dataset(self):
        train_length = int(len(self.image_set)*self.train_test_ratio)
        train_imgs = self.image_set[0:train_length]
        return ImageDataLoader.load_dataset(train_imgs, self.data_config)
    
    def get_test_dataset(self):
        train_length = int(len(self.image_set)*self.train_test_ratio)
        test_imgs = self.image_set[train_length:]
        return ImageDataLoader.load_dataset(test_imgs, self.data_config)
