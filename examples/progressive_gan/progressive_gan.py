import os
from typing import Tuple
from time import time

import numpy as np
import tensorflow as tf
from karys.data.ImageDataLoader import ImageDataLoader
from keras.losses import CategoricalCrossentropy, MeanSquaredError
from keras.models import load_model, Model
from keras.layers import Input
from keras.optimizers import Adam
from karys.layers.LearnedNoise import LearnedNoise
from karys.layers.MinibatchDiscrimination import MinibatchDiscrimination
from karys.layers.WeightedAdd import WeightedAdd
from karys.layers.AdaptiveInstanceNormalization import AutoAdaptiveInstanceNormalization, AdaptiveInstanceNormalization
from karys.models.bases import GraphableModelBlock
from karys.models.convolutional_blocks import Conv2DBatchNormLeakyReluBlock, Conv2DNormActBlock, LayerNormActBlock
from karys.models.passthrough_blocks import PassthroughBlock, PassthroughLeanredNoiseBlock
# from karys.layers.MeanSquaredError import MeanSquaredError
from karys.trainers.ProgressiveGanTrainer import ProgressiveGanTrainer
from karys.models.progressive_discriminator import ProgressiveDiscriminator224x224, ProgressiveDiscriminator112x112, ProgressiveDiscriminator56x56, ProgressiveDiscriminator28x28, ProgressiveDiscriminator14x14, ProgressiveDiscriminator7x7, ProgressivePassthroughDescriminator
from karys.models.progressive_generator import ProgressiveGenerator112x112, ProgressiveGenerator14x14, ProgressiveGenerator224x224, ProgressiveGenerator28x28, ProgressiveGenerator56x56, ProgressiveGenerator7x7, ProgressivePassthroughGenerator
CUSTOM_OBJECTS={
                'WeightedAdd': WeightedAdd,
                'PassthroughBlock': PassthroughBlock,
                'GraphableModelBlock': GraphableModelBlock,
                'Conv2DNormActBlock': Conv2DNormActBlock,
                'LayerNormActBlock': LayerNormActBlock,
                'MinibatchDiscrimination': MinibatchDiscrimination,
                'Conv2DBatchNormLeakyReluBlock': Conv2DBatchNormLeakyReluBlock,
                'AutoAdaptiveInstanceNormalization': AutoAdaptiveInstanceNormalization,
                'AdaptiveInstanceNormalization': AdaptiveInstanceNormalization,
                'ProgressiveDiscriminator224x224': ProgressiveDiscriminator224x224,
                'ProgressivePassthroughDescriminator': ProgressivePassthroughDescriminator,
                'ProgressiveDiscriminator112x112': ProgressiveDiscriminator112x112,
                'ProgressiveDiscriminator56x56': ProgressiveDiscriminator56x56,
                'ProgressiveDiscriminator28x28': ProgressiveDiscriminator28x28,
                'ProgressiveDiscriminator14x14': ProgressiveDiscriminator14x14,
                'ProgressiveDiscriminator7x7': ProgressiveDiscriminator7x7,
                'ProgressiveGenerator224x224': ProgressiveGenerator224x224,
                'ProgressivePassthroughGenerator': ProgressivePassthroughGenerator,
                'PassthroughLeanredNoiseBlock': PassthroughLeanredNoiseBlock,
                'ProgressiveGenerator112x112': ProgressiveGenerator112x112,
                'ProgressiveGenerator56x56': ProgressiveGenerator56x56,
                'ProgressiveGenerator28x28': ProgressiveGenerator28x28,
                'ProgressiveGenerator14x14': ProgressiveGenerator14x14,
                'ProgressiveGenerator7x7': ProgressiveGenerator7x7,
            }
input_path = "C:/Users/pdkar/dev/Datasets/Handwritten"
output_path = "./examples/progressive_gan/test_output"

gen_path = output_path + "/progressive_generator"
disc_path = output_path + "/progressive_discriminator"

gen_optimizer = Adam(learning_rate=1e-4)
disc_optimizer = Adam(learning_rate=1e-5)

noise_size = 256
feature_size = 256

style_loss_fn = MeanSquaredError()
gen_loss_fn = CategoricalCrossentropy()
disc_loss_fn = CategoricalCrossentropy()

tensorboard_dir = output_path + "/gan_board_logs"
tensorboard_writer = tf.summary.create_file_writer(tensorboard_dir)
batch_size = 4

def load_models() -> Tuple[Model, Model, ProgressiveGanTrainer]:
    if not os.path.exists(output_path + "/architecture_diagrams"):
        os.mkdir(output_path + "/architecture_diagrams")
    if not os.path.exists(output_path + "/architecture_diagrams"):
        os.mkdir(output_path + "/architecture_diagrams")
    global input_path, gen_optimizer, gen_loss_fn, disc_optimizer, disc_loss_fn, noise_size
    data_loader = ImageDataLoader(input_path + "/", ".jpg", 0.10, (28, 28))
    num_categories = len(data_loader.label_set)
    # try:
    print("loading classifier...")
    # discriminator28 = load_model(disc_path, CUSTOM_OBJECTS)
    # discriminator = discriminator28
    # discriminator = ProgressiveDiscriminator14x14(feature_size=feature_size, category_size=num_categories)#,next_discriminator=discriminator28)
    discriminator = ProgressiveDiscriminator28x28(feature_size=feature_size, category_size=num_categories)
    discriminator.plot_graphable_model(output_path + '/architecture_diagrams')
    discriminator = discriminator.build_graph()
    discriminator.summary()

    # try:
    print("loading generator...")
    # generator28 = load_model(gen_path, CUSTOM_OBJECTS)
    # generator = generator28
    # generator = ProgressiveGenerator14x14(noise_size, num_categories)
    generator = ProgressiveGenerator28x28(noise_size, num_categories)#, previous_generator=generator14)
    # generator = ProgressiveGenerator56x56(latent_noise_size=noise_size, previous_generator=generator28)
    generator.plot_graphable_model(output_path + '/architecture_diagrams')
    generator = generator.build_graph()
    generator.summary()
    trainer = ProgressiveGanTrainer(generator, discriminator, gen_optimizer,
                                    gen_loss_fn, disc_optimizer, disc_loss_fn,
                                    style_loss_fn, data_loader,
                                    tensorboard_writer)
    return generator, discriminator, trainer


def train(epochs, trains_per_test):
    global disc_path
    generator, discriminator, trainer = load_models()
    if not os.path.exists(output_path + "/disc_real"):
        os.mkdir(output_path + "/disc_real")
    if not os.path.exists(output_path + "/disc_gen"):
        os.mkdir(output_path + "/disc_gen")
    if not os.path.exists(output_path + "/generated"):
        os.mkdir(output_path + "/generated")
    # cols=["epoch", "gen_loss", "style_loss", "disc_loss", "time_step"]

    # data_file = pd.read_csv(output_path + "/datalog.csv") if os.path.exists(output_path + "/datalog.csv") else pd.DataFrame(columns=cols)

    for i in range(epochs):
        # start_time = time()

        if i % trains_per_test == 0 and i != 0:
            gen_loss, style_loss, disc_loss = trainer.test(batch_size, 1)
            disc_real_output_filename = output_path + "/disc_real/" + str(i) + ".jpg"
            disc_gen_output_filename = output_path + "/disc_gen/" + str(i) + ".jpg"
            gen_output_filename = output_path + "/generated/" + str(i) + ".jpg"
            trainer.save_progressive_disc_real_images(disc_real_output_filename)
            trainer.save_progressive_disc_gen_images(disc_gen_output_filename)
            trainer.save_progressive_gen_images(gen_output_filename)
        else:
            gen_loss, style_loss, disc_loss = trainer.train(batch_size, 6)

        # end_time = time()
        # test_label = "TEST" if i % trains_per_test == 0 and i != 0 else ""
        # dr = [i, gen_loss, style_loss, disc_loss, end_time - start_time]
        # data_row = pd.DataFrame([dr], columns=cols)
        # data_file = pd.concat([data_file, data_row], ignore_index=True)
        # print(f"EPOCH {i}/{epochs}:\tgen loss={gen_loss}\tstyle loss={style_loss}\tdisc loss={disc_loss}\ttime={end_time-start_time}\t{test_label}")
        # data_file.to_csv(output_path + "/datalog.csv")
    discriminator.save(disc_path)
    generator.save(gen_path)