from matplotlib import pyplot as plt
import numpy as np
from PIL import Image

from typing import List
from data.configs.ImageDataConfig import ImageDataConfig

from data.wrappers.DataWrapper import DataWrapper
import glob


def convert_image(img, channels):
    if channels == 4:
        return img.convert('RGBA')
    elif channels == 3:
        return img.convert('RGB')
    elif channels == 1:
        return img.convert('L')
    else:
        raise ValueError("Channels must be one of [1,3,4]")

def load_images(dirpath: str, data_ref: ImageDataConfig):
    glob_glob = dirpath + "/*" + data_ref.image_type
    images = glob.glob(glob_glob)
    print("LOADING FROM %s" % (glob_glob))
    print("LOADING %d IMAGES" % len(images))
    x = []
    num_images = len(images)
    for n, i in enumerate(images):
        if 100*n/num_images >= data_ref.load_n_percent:
            break
        x.append(Image.open(i))
    print("LOADED %d IMAGES" % len(x))
    return x

class ImageDataWrapper(DataWrapper):
    def __init__(self,image_set: List, data_config: ImageDataConfig, validation_percentage: float = 0.05):
        super(ImageDataWrapper, self).__init__(data_config, validation_percentage)
        self.image_set: np.ndarray = image_set
        self.data_config = data_config
    
    @classmethod
    def load_from_file(cls,filename, data_config: ImageDataConfig, validation_percentage: float = 0.05):
        images = load_images(filename, data_config)

        img_rows, img_cols, channels = data_config.image_shape
        imgs = []
        for img in images:
            img = convert_image(img,channels)
            img = img.resize(size=(img_rows, img_cols),resample=Image.ANTIALIAS)
            img = np.array(img).astype('float32')
            img = data_config.load_scale_func(img)
            imgs.append(img)
        return cls(np.array(imgs), data_config, validation_percentage)
    
    def get_train_dataset(self):
        validation_length = int(len(self.image_set)*self.validation_percentage)
        return self.image_set[:-validation_length]
    
    def get_validation_dataset(self):
        validation_length = int(len(self.image_set)*self.validation_percentage)
        return self.image_set[-validation_length:]
    
    def save_generated_images(self, filename, generated_images):
        image_count = 0
        image_shape = generated_images.shape[-3:]
        img_size = image_shape[1]
        channels = image_shape[-1]
        preview_height = self.data_config.preview_rows*img_size + (self.data_config.preview_rows + 1)*self.data_config.preview_margin
        preview_width = self.data_config.preview_cols*img_size + (self.data_config.preview_cols + 1)*self.data_config.preview_margin
        
        if channels ==1:
            image_array = np.full((preview_height, preview_width), 255, dtype=np.uint8)
        else:
            image_array = np.full((preview_height, preview_width, channels), 255, dtype=np.uint8)
        for row in range(self.data_config.preview_rows):
            for col in range(self.data_config.preview_cols):
                r = row * (img_size+self.data_config.preview_margin) + self.data_config.preview_margin
                c = col * (img_size+self.data_config.preview_margin) + self.data_config.preview_margin
                img = generated_images[image_count]
                img = self.data_config.save_scale_func(img)
                if channels == 1:
                    img = np.reshape(img,newshape=(img_size,img_size))
                else:
                    img = np.array(img)
                    img = Image.fromarray((img).astype(np.uint8))
                    img = img.resize((img_size,img_size),Image.BICUBIC)
                    img = np.asarray(img)
                    
                image_array[r:r+img_size, c:c+img_size] = img
                image_count += 1

        im = Image.fromarray(image_array.astype(np.uint8))
        im.save(filename)
    
    def save_classified_images(self, filename, target_images_with_labels, noise_imaegs_with_labels):
        image_shape = target_images_with_labels[0][0].shape
        img_size = image_shape[1]
        channels = image_shape[-1]
        preview_height = self.data_config.preview_rows*img_size + (self.data_config.preview_rows + 1)*self.data_config.preview_margin
        preview_width = self.data_config.preview_cols*img_size + (self.data_config.preview_cols + 1)*self.data_config.preview_margin
        
        fig,axes = plt.subplots(self.data_config.preview_rows,self.data_config.preview_cols)
        fig.set_figheight(preview_height)
        fig.set_figwidth(preview_width)

        for row in range(self.data_config.preview_rows):
            for col in range(self.data_config.preview_cols):
                use_target = col % 2 == 0

                img, label = target_images_with_labels[row+col] if use_target else noise_imaegs_with_labels[row+col]
                
                max_label = np.argmax(label)
                out_label = "HOT DOG" if max_label == 0 else "NOT HOT DOG"
                pass_fail = "PASS" if (max_label == 0 and use_target) or (max_label == 1 and not use_target) else "FAIL"

                text_label = out_label + " || " + pass_fail
                img = self.data_config.save_scale_func(img)

                if channels == 1:
                    img = np.reshape(img,newshape=(img_size,img_size))
                else:
                    img = np.array(img)
                    img = Image.fromarray((img).astype(np.uint8))
                    img = img.resize((img_size,img_size),Image.BICUBIC)
                    img = np.asarray(img)
                    
                axes[row,col].imshow(img)
                axes[row,col].set_title(text_label, fontsize=img_size*2)

        fig.savefig(filename)
        plt.close()

    
