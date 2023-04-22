import os
from typing import Tuple
from time import time

import tensorflow as tf
from karys.data.ImageDataLoader import ImageDataLoader
from keras.losses import BinaryCrossentropy, MeanSquaredError, MeanAbsoluteError, CategoricalCrossentropy
from keras.models import load_model, Model
from keras.layers import Input
from keras.optimizers import Adam
from karys.layers.AdaptiveInstanceNormalization import AdaptiveInstanceNormalization, AutoAdaptiveInstanceNormalization
from karys.layers.WeightedAdd import WeightedAdd
from karys.models.bases import GraphableModelBlock
from karys.models.convolutional_blocks import Conv2DBatchNormLeakyReluBlock, Conv2DNormActBlock, LayerNormActBlock
from karys.models.passthrough_blocks import PassthroughBlock
# from karys.layers.MeanSquaredError import MeanSquaredError
from karys.trainers.ProgressiveDiscriminatorTrainer import ProgressiveDiscriminatorTrainer
from karys.models.progressive_discriminator import ProgressiveDiscriminator112x112, ProgressiveDiscriminator14x14, ProgressiveDiscriminator224x224, ProgressiveDiscriminator28x28, ProgressiveDiscriminator56x56, ProgressiveDiscriminator7x7, ProgressivePassthroughDescriminator

base_path = "./examples/progressive_discriminator"
input_path = "C:/Users/pdkar/dev/Datasets/Handwritten"
output_path = "./examples/progressive_gan/test_output"

disc_path = output_path + "/progressive_discriminator"

disc_optimizer = Adam(learning_rate=8e-4)

feature_size = 128

style_loss_fn = MeanAbsoluteError(reduction="sum")
disc_loss_fn = CategoricalCrossentropy(from_logits=True)

tensorboard_writer = tf.summary.create_file_writer(output_path + "/board_logs")

def load_models() -> Tuple[Model, Model, ProgressiveDiscriminatorTrainer]:
    global input_path, disc_optimizer, disc_loss_fn
    data_loader = ImageDataLoader(input_path + "/",".jpg",0.1)
    num_categories = len(data_loader.label_set)
    try:
        print("loading classifier...")
        discriminator = load_model(disc_path,
            custom_objects={
                'ProgressiveDiscriminator224x224': ProgressiveDiscriminator224x224,
                'ProgressivePassthroughDescriminator': ProgressivePassthroughDescriminator,
                'PassthroughBlock': PassthroughBlock,
                'GraphableModelBlock': GraphableModelBlock,
                'ProgressiveDiscriminator112x112': ProgressiveDiscriminator112x112,
                'ProgressiveDiscriminator56x56': ProgressiveDiscriminator56x56,
                'ProgressiveDiscriminator28x28': ProgressiveDiscriminator28x28,
                'ProgressiveDiscriminator14x14': ProgressiveDiscriminator14x14,
                'ProgressiveDiscriminator7x7': ProgressiveDiscriminator7x7,
                'Conv2DBatchNormLeakyReluBlock': Conv2DBatchNormLeakyReluBlock,
                'AutoAdaptiveInstanceNormalization': AutoAdaptiveInstanceNormalization,
                'AdaptiveInstanceNormalization': AdaptiveInstanceNormalization,
                'Conv2DNormActBlock': Conv2DNormActBlock,
                'LayerNormActBlock': LayerNormActBlock,
                'WeightedAdd': WeightedAdd
            })
    except:
        print("failed to load classifier... building")
        discriminator = ProgressiveDiscriminator224x224(feature_size, num_categories)
        if not os.path.exists(output_path + "/architecture_diagrams"):
            os.mkdir(output_path + "/architecture_diagrams")
        discriminator.plot_graphable_model(output_path + "/architecture_diagrams")
        discriminator = discriminator.build_graph()
    discriminator.summary()
    
    trainer = ProgressiveDiscriminatorTrainer(discriminator, Adam(), CategoricalCrossentropy(), style_loss_fn, data_loader, tensorboard_writer)
    return discriminator, trainer

def train(epochs, trains_per_test):
    discriminator, trainer = load_models()
    
    for i in range(epochs):
        start_time = time()

        if i % trains_per_test == 0 and i != 0:
            disc_loss = trainer.test(3,1)
            real_output_filename = output_path + "/real-" + str(i) + ".jpg"
            trainer.save_progressive_real_images(real_output_filename)
        else:
            disc_loss = trainer.train(4, 3)
        
        end_time = time()
        test_label = "TEST" if i % trains_per_test == 0 and i != 0 else ""
        print(f"EPOCH {i}/{epochs}: disc loss={disc_loss}, time={end_time-start_time}, {test_label}")
    discriminator.save(disc_path)