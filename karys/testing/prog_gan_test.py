# from karys.data.configs.ImageDataConfig import ImageDataConfig
# from karys.data.configs.RandomDataConfig import RandomDataConfig
# from karys.data.wrappers.ImageDataWrapper import ImageDataWrapper
# from karys.data.wrappers.RandomDataWrapper import RandomDataWrapper
# # from karys.models.progressive_discriminator import ProgressiveDiscriminator
# from karys.models.progressive_generator import ProgressiveGenerator
# from karys.trainers.ProgressiveGanTrainer import ProgressiveGanTrainer

# from keras.losses import BinaryCrossentropy, MeanSquaredError
# from keras.optimizers import Adam
# from time import time
# import numpy as np

# def test_build_prog_gan(epochs, trains_per_test):

#     ##set up data
#     base_path = "./examples/discriminator/test_input/Fruit"
#     extra_labels = ["generated"]
#     image_config = ImageDataConfig(image_shape=(224,224, 3),image_type=".jpg", preview_rows=4, preview_cols=4, load_n_percent=20)
#     image_data_wrapper = ImageDataWrapper.load_from_labelled_directories(base_path + '/', image_config, extra_labels, validation_percentage=0.1)

#     random_config = RandomDataConfig([1024], 0.0, 1.0)
#     random_data_wrapper = RandomDataWrapper(random_config)

#     ##set up models
#     # disc = ProgressiveDiscriminator(4096)
    
#     disc_graph = disc.build_graph()
#     print(disc_graph.input_shape)
#     print(disc_graph.output_shape)

#     gen = ProgressiveGenerator(512)
    
#     gen_graph = gen.build_graph()
#     print(gen_graph.input_shape)
#     print(gen_graph.output_shape)

#     ##set up training
#     gen_optimizer = Adam(learning_rate=4e-5)
#     gen_loss = BinaryCrossentropy(from_logits=True,reduction="sum_over_batch_size")
    
#     disc_optimizer = Adam(learning_rate=4e-5)
#     disc_loss = BinaryCrossentropy(from_logits=True,reduction="sum_over_batch_size")

#     style_loss = MeanSquaredError(reduction="sum_over_batch_size")
#     trainer = ProgressiveGanTrainer(gen,disc,gen_optimizer,gen_loss,disc_optimizer,disc_loss,style_loss,random_data_wrapper,image_data_wrapper)

#     test_loss = 0
#     for i in range(epochs):
#         start_time = time()
#         train_loss = trainer.train(4, 1)
#         avg_loss = np.mean(train_loss)

#         if i % trains_per_test == 0 and i != 0:
#             test_loss = trainer.test(16,1)
#             avg_loss = np.mean(test_loss)
#             test_output_filename = output_path + "/train-" + str(i) + ".jpg"
#             data_wrapper.save_classified_images(test_output_filename, trainer.most_recent_output, img_size=32)
        
#         end_time = time()
#         print(f"EPOCH {i}/{epochs}: loss={avg_loss}, time={end_time-start_time}")


