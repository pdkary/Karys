from karys.data.ImageDataLoader import ImageDataLoader
import numpy as np
def test_load_data():
    test_dir = "./examples/discriminator/test_input/Fruit"
    dl = ImageDataLoader(test_dir,".jpg")

    N,B = 10,4
    batches = dl.get_sized_training_batch_set(N,B,(224,224))
    for b in batches:
        print("[")
        for lbl, img in b:
            print("\t",lbl, img.shape,",")
        print("]")
    