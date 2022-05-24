import glob
from PIL import Image
import numpy as np
import tensorflow as tf

from data.configs.ImageDataConfig import ImageDataConfig



def load_images(dirpath: str, image_type:str, load_n_percent=100):
    glob_glob = dirpath + "/*" + image_type
    images = glob.glob(glob_glob)
    print("LOADING FROM %s" % (glob_glob))
    print("LOADING %d IMAGES" % len(images))
    x = []
    num_images = len(images)
    for n, i in enumerate(images):
        if 100*n/num_images >= load_n_percent:
            break
        x.append(Image.open(i))
    print("LOADED %d IMAGES" % len(x))
    return x

def load_dataset(imageset, data_ref: ImageDataConfig):
    img_rows, img_cols, channels = data_ref.image_shape
    imgs = []
    for img in imageset:
        if channels == 4:
            img = img.convert('RGBA')
        elif channels == 3:
            img = img.convert('RGB')
        elif channels == 1:
            img = img.convert('L')
        img = img.resize(size=(img_rows, img_cols),resample=Image.ANTIALIAS)
        img = np.array(img).astype('float32')
        img = data_ref.load_scale_func(img)
        imgs.append(img)
    return tf.data.Dataset.from_tensor_slices((np.array(imgs),np.ones((len(imgs))))).batch(data_ref.batch_size)
    
def save_images(filename, generated_images, data_ref: ImageDataConfig):
    image_count = 0
    image_shape = generated_images.shape[-3:]
    img_size = image_shape[1]
    channels = image_shape[-1]
    preview_height = data_ref.preview_rows*img_size + (data_ref.preview_rows + 1)*data_ref.preview_margin
    preview_width = data_ref.preview_cols*img_size + (data_ref.preview_cols + 1)*data_ref.preview_margin
    
    if channels ==1:
        image_array = np.full((preview_height, preview_width), 255, dtype=np.uint8)
    else:
        image_array = np.full((preview_height, preview_width, channels), 255, dtype=np.uint8)
    for row in range(data_ref.preview_rows):
        for col in range(data_ref.preview_cols):
            r = row * (img_size+data_ref.preview_margin) + data_ref.preview_margin
            c = col * (img_size+data_ref.preview_margin) + data_ref.preview_margin
            img = generated_images[image_count]
            img = data_ref.save_scale_func(img)
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