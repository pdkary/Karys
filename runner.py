from karys.testing import classifier_test, image_data_wrapper_test, gan_test, prog_gan_test
from examples.generative_adversarial import gan
from examples.discriminator import classifier

# image_data_wrapper_test.test_load_from_labelled_directories()
# model_builder_test.generator_model_test()
# classifier_test.build()
# gan_test.test_gan_model()
classifier.train(1000,10)

# prog_gan_test.test_build_prog_gan()