import re
from matplotlib import pyplot as plt
import numpy as np
from PIL import Image
import json
import os

from typing import Dict
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

def load_labels(filepath):
    labels = {}
    with open(filepath, 'r') as f:
        labels = json.load(f)
    return labels

def load_images(dirpath: str, data_ref: ImageDataConfig):
    glob_glob = dirpath + "**/*" + data_ref.image_type
    images = glob.glob(glob_glob)
    np.random.shuffle(images)

    print("LOADING FROM %s" % (glob_glob))
    print("LOADING %d IMAGES" % len(images))
    x = {}
    num_images = len(images)

    ten_percent = data_ref.load_n_percent/10
    prev_progress = 0
    for n, i in enumerate(images):
        current_state = 100*n/num_images
        current_progress = current_state // ten_percent
        if current_progress > prev_progress:
            print(f"Loaded {round(current_state,2)} %")
            prev_progress = current_progress
        
        if current_state >= data_ref.load_n_percent:
            break
        image_name = re.split("[\\\/]+",i)[-1]
        x[image_name] = Image.open(i)
    print("LOADED %d IMAGES" % len(x))
    return x

class ImageDataWrapper(DataWrapper):
    def __init__(self, 
                 image_set: Dict[str, np.ndarray], 
                 image_labels: Dict[str,str],
                 data_config: ImageDataConfig, 
                 validation_percentage: float = 0.05):
        super(ImageDataWrapper, self).__init__(data_config, validation_percentage)
        self.image_set = image_set
        self.image_labels = {x:image_labels[x] for x in image_set.keys()}
        self.data_config = data_config
    
    @classmethod
    def load_from_file_with_single_label(cls, image_filepath, label, data_config: ImageDataConfig, validation_percentage: float = 0.05):
        glob_glob = image_filepath + "**/*" + data_config.image_type
        images = glob.glob(glob_glob)
        labels = {re.split("[\\\/]+",i)[-1]:label for i in images}
        return ImageDataWrapper.load_from_file(image_filepath, labels, data_config, validation_percentage)

    @classmethod
    def load_from_files(cls, image_filepath, label_filepath, data_config: ImageDataConfig, validation_percentage: float = 0.05):
        labels = load_labels(label_filepath)
        return ImageDataWrapper.load_from_file(image_filepath, labels, data_config, validation_percentage)

    @classmethod
    def load_from_labelled_directories(cls, base_dir, data_config: ImageDataConfig, validation_percentage: float = 0.05, use_dirs=None):
        directory_glob = glob.glob(base_dir + '/*/')
        label_dict = {}
        for folder in directory_glob:
            label = re.split("[\\\/]+",folder)[-2]
            if use_dirs is None or label in use_dirs:
                for filepath in glob.glob(folder+"/*"):
                    filename = os.path.basename(filepath)
                    label_dict[filename] = label
            else:
                print("skipping: ", label)
        return ImageDataWrapper.load_from_file(base_dir, label_dict, data_config, validation_percentage)

    @classmethod
    def load_from_file(cls, image_filepath, labels: Dict[str,str], data_config: ImageDataConfig, validation_percentage: float = 0.05):
        images = load_images(image_filepath, data_config)
        img_rows, img_cols, channels = data_config.image_shape
        imgs = {}
        for img_name in images.keys():
            if img_name in labels.keys():
                img = images[img_name]
                img = convert_image(img,channels)
                img = img.resize(size=(img_rows, img_cols),resample=Image.ANTIALIAS)
                img = np.array(img).astype('float32')
                img = data_config.load_scale_func(img)
                imgs[img_name] = img
        return cls(imgs, labels, data_config, validation_percentage)

    
    def get_train_dataset(self) -> Dict[str, np.ndarray]:
        validation_length = int(len(self.image_set)*self.validation_percentage)
        train_keys = list(self.image_set.keys())[:-validation_length]
        return {k:self.image_set[k] for k in train_keys}
    
    def get_validation_dataset(self) -> Dict[str, np.ndarray]:
        validation_length = int(len(self.image_set)*self.validation_percentage)
        validation_keys = list(self.image_set.keys())[-validation_length:]
        return {k:self.image_set[k] for k in validation_keys}
    
    def save_generated_images(self, filename, generated_images):
        image_count = 0
        image_shape = generated_images.shape[-3:]
        img_size = image_shape[1]
        channels = image_shape[-1]

        R, C, M = self.data_config.preview_rows, self.data_config.preview_cols, self.data_config.preview_margin
        
        preview_height = R*img_size + (R + 1)*M
        preview_width = C*img_size + (C + 1)*M
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
    
    def save_classified_images(self, filename, images_with_labels_and_preds, img_size = 32):
        image_shape = images_with_labels_and_preds[0][0].shape
        channels = image_shape[-1]

        R, C, M = self.data_config.preview_rows, self.data_config.preview_cols, self.data_config.preview_margin
        
        preview_height = R*img_size + (R + 1)*M
        preview_width = C*img_size + (C + 1)*M
        
        fig,axes = plt.subplots(self.data_config.preview_rows,self.data_config.preview_cols)
        fig.set_figheight(preview_height)
        fig.set_figwidth(preview_width)

        for row in range(R):
            for col in range(C):
                img, label, pred = images_with_labels_and_preds[row*R+col]
                pass_fail = "PASS" if np.all(pred == label) else "FAIL"

                text_label = pred + " || " + label + " || " + pass_fail
                img = self.data_config.save_scale_func(img)

                if channels == 1:
                    img = np.reshape(img,newshape=(img_size,img_size))
                else:
                    img = np.array(img)
                    img = Image.fromarray((img).astype(np.uint8))
                    img = img.resize((img_size,img_size), Image.BICUBIC)
                    img = np.asarray(img)
                    
                axes[row,col].imshow(img)
                axes[row,col].set_title(text_label, fontsize=img_size*2)

        fig.savefig(filename)
        plt.close()
    
    def save_encoded_images(self, filename, images_labels_encodings, img_size = 32):
        image_shape = images_labels_encodings[0][0].shape
        encoding_dim = images_labels_encodings[0][2].shape[-1]
        channels = image_shape[-1]

        R, C, M = self.data_config.preview_rows, self.data_config.preview_cols, self.data_config.preview_margin
        
        preview_height = R*img_size + (R + 1)*M
        preview_width = C*img_size + (C + 1)*M
        
        fig,axes = plt.subplots(R,C*2)
        fig.set_figheight(preview_height)
        fig.set_figwidth(preview_width)

        for row in range(R):
            for col in range(C):
                img, label, encoding  = images_labels_encodings[row*R+col]
                img = self.data_config.save_scale_func(img)

                if channels == 1:
                    img = np.reshape(img,newshape=(img_size,img_size))
                else:
                    img = np.array(img)
                    img = Image.fromarray((img).astype(np.uint8))
                    img = img.resize((img_size,img_size), Image.BICUBIC)
                    img = np.asarray(img)
                    
                axes[row,2*col].imshow(img)
                axes[row,2*col+1].bar(range(encoding_dim), encoding)
                axes[row,2*col].set_title(label, fontsize=img_size*2)
                axes[row,2*col+1].set_title(label, fontsize=img_size*2)

        fig.savefig(filename)
        plt.close()

    
