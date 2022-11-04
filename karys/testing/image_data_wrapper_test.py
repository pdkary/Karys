from karys.data.configs.ImageDataConfig import ImageDataConfig
from karys.data.wrappers.ImageDataWrapper import ImageDataWrapper


def test_load_from_labelled_directories():
##set up data
    base_path = "./examples/discriminator/test_input/Fruit"
    extra_labels = ["generated"]
    image_config = ImageDataConfig(image_shape=(224,224, 3),image_type=".jpg", preview_rows=4, preview_cols=4, load_n_percent=20)
    data_wrapper = ImageDataWrapper.load_from_labelled_directories(base_path + '/', image_config, extra_labels, validation_percentage=0.1)

    data_wrapper.summary()
    orange_vec = data_wrapper.label_vectorizer.get_label_vector_by_category_name("Orange")
    print(orange_vec)