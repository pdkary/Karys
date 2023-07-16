import glob
from PIL import Image
import numpy as np
import tensorflow as tf
import random
from copy import deepcopy

from karys.data.labels.CategoricalLabel import CategoricalLabel

def scale_0_255(img):
    return 255*scale_0_1(img)

def scale_0_1(img):
    eps = 1e-8
    return (img - np.min(img))/(np.max(img) - np.min(img) + eps)

def resize(img, newsize):
    img = np.asarray(img,dtype=np.float32)
    if not img.shape[:-1] == newsize:
        img = scale_0_255(img).astype(np.uint8)
        img = Image.fromarray(img)
        img = img.resize(newsize,Image.BICUBIC)
        img = np.asarray(img,dtype=np.float32)
        return scale_0_1(img)
    return img

#ok so we want to load images from a directory, and hold them to use later, at variable sizes
class ImageDataLoader():
    

    ##base directory should contain subfolders, labelled by the type of image contained
    ##ie) Fruit:
    ##      Apple:
    ##          Apple1.jpg
    ##          Apple2.jpg
    ##      Orange
    ##          Orange.jpg
    def __init__(self, base_directory, image_type, validation_percentage=0.1, load_size=None):
        self.train_images, self.validation_images = self.load_images(base_directory, image_type, validation_percentage, load_size=load_size)
        self.summary()
        self.label_vectorizer: CategoricalLabel = CategoricalLabel(self.label_set)

    @property
    def label_set(self):
        return list(self.train_images.keys())
        # return [*self.train_images.keys(), "invalid"]
    
    # @property
    # def invalid_flag(self):
    #     return self.label_set[-1]
    
    @property
    def all_images(self):
        all_images = self.train_images
        for k,v in self.validation_images.items():
            all_images[k].extend(v)
        return all_images
    
    def summary(self):
        print("------BEGIN DATA SUMMARY------------")
        print("--\t------TRAIN---------------------")
        for key, val in self.train_images.items():
            if len(val) > 0:
                space = max(12 - len(key),0)
                print(f"\t{key}{' '*space}\t|\t{len(val)}\t|\t{val[0].shape}")
        print("--\t------TEST----------------------")
        for key, val in self.validation_images.items():
            if len(val) > 0:
                space = max(12 - len(key),0)
                print(f"\t{key}{' '*space}\t|\t{len(val)}\t|\t{val[0].shape}")
        print("------END DATA SUMMARY--------------")
    
    def get_all_images_sized(self, newsize):
        imgs = self.all_images
        output = {k:[] for k in imgs.keys()}
        for k,v in imgs.items():
            for img in v:
                output[k].append(resize(img,newsize))
        return output

    def get_label_vectors_by_names(self, batch_labels):
        return self.label_vectorizer.get_label_vectors_by_category_names(batch_labels)
    
    def get_label_names_by_ids(self, classification_ids):
        return self.label_vectorizer.get_label_names_by_ids(classification_ids)
    
    def get_label_vectors_by_ids(self, classification_ids):
        return self.label_vectorizer.get_label_vectors_by_ids(classification_ids)

    def load_images(self, base_directory, image_type, validation_percentage, load_size=None):
        train_images, validation_images = {},{}
        glob_glob = base_directory + "/*" 
        image_sets = glob.glob(glob_glob)
        for image_set in image_sets:
            set_name = image_set.replace(base_directory[:-1] + '\\','')
            train_images[set_name], validation_images[set_name] = [], []
            for image in glob.glob(image_set + "/*"+image_type):
                tmp = Image.open(image)
                if load_size is not None:
                    tmp = tmp.resize(load_size)
                tmp_cpy = np.array(tmp.copy(),dtype=np.float32)
                tmp_cpy = (tmp_cpy - np.min(tmp_cpy))/(np.max(tmp_cpy) - np.min(tmp_cpy) + 1e-8)
                if random.uniform(0,1) > validation_percentage:
                    train_images[set_name].append(tmp_cpy)
                else:
                    validation_images[set_name].append(tmp_cpy)
                tmp.close()
            print(f"Finished loading {set_name}s")
        return train_images, validation_images


    def get_train_tuples(self):
        return [(k,v) for k in self.train_images.keys() for v in self.train_images[k]]
    
    def get_validation_tuples(self):
        return [(k,v) for k in self.validation_images.keys() for v in self.validation_images[k]]
    
    def get_sized_training_batch_set(self, num_batches, batch_size, image_size):
        train_tuples = self.get_train_tuples()
        return self.__get_sized_batch_set__(train_tuples, num_batches, batch_size, image_size)
    
    def get_sized_validation_batch_set(self, num_batches, batch_size, image_size):
        validation_tuples = self.get_validation_tuples()
        return self.__get_sized_batch_set__(validation_tuples, num_batches, batch_size, image_size)
    
    def get_multi_sized_training_batch_set(self, num_batches, batch_size, image_sizes):
        train_tuples = self.get_train_tuples()
        return self.__get_multi_sized_batch_set__(train_tuples, num_batches, batch_size, image_sizes)
    
    def get_multi_sized_validation_batch_set(self, num_batches, batch_size, image_sizes):
        validation_tuples = self.get_validation_tuples()
        return self.__get_multi_sized_batch_set__(validation_tuples, num_batches, batch_size, image_sizes)
    
    def __get_multi_sized_batch_set__(self, tuple_set, num_batches, batch_size, image_sizes):
        output = []

        for batch_inds in np.random.choice(len(tuple_set), size=(num_batches, batch_size)):
            batch_labels, batch_data = [], {i:[] for i in image_sizes}
            for ind in batch_inds:
                lbl,img = tuple_set[ind]
                for image_size in image_sizes:
                    img = resize(img, image_size)
                    batch_data[image_size].append(img)

                batch_labels.append(lbl)
            
            batch_data = [tf.stack(v) for v in batch_data.values()]
            lbl_vectors = self.label_vectorizer.get_label_vectors_by_category_names(batch_labels)
            flag_vectors = self.label_vectorizer.get_label_vectors_by_category_names([self.invalid_flag]*batch_size)
            output.append((batch_labels, lbl_vectors, flag_vectors, *batch_data))
        return output
    
    def __get_sized_batch_set__(self, tuple_set, num_batches, batch_size, image_size):
        output = []
        N = num_batches
        B = batch_size
        categories = deepcopy(self.label_vectorizer.categories)
        # categories.remove("invalid")
        #we want to get N batches of size B, with each batch containing values of all the same label
        cats = np.random.choice(categories, N)
        for cat in cats:
            batch_labels, batch_data = [], []
            cat_tuples = [(k,v) for k,v in tuple_set if k == cat]
            #now we want to grab B of these tuples for each batch
            selected_inds = np.random.choice(len(cat_tuples), B)
            for ind in selected_inds:
                lbl, img = cat_tuples[ind]
                img = resize(img, image_size)
                batch_labels.append(lbl)
                batch_data.append(img)
            lbl_vectors = self.label_vectorizer.get_label_vectors_by_category_names(batch_labels)
            # flag_vectors = self.label_vectorizer.get_label_vectors_by_category_names([self.invalid_flag]*batch_size)
            # output.append((batch_labels, lbl_vectors, flag_vectors, np.array(batch_data, dtype=np.float32)))
            output.append((batch_labels, lbl_vectors, np.array(batch_data, dtype=np.float32)))

        return output

        
    
    
    
            