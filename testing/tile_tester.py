from image_utils import ImageTiler
from PIL import Image

img = Image.open('painting.jpg')
H,W = img.size

tiles = ImageTiler.get_tiled_image(img, 50, 50)
print(tiles.shape)

retiled_image = ImageTiler.retile_image(tiles)
print(retiled_image.size)
retiled_image.show()

hilbert_stack = ImageTiler.get_hilbert_tile_stack(img, 50, 50)
print(hilbert_stack.shape)

retiled_hilbert_image = ImageTiler.retile_hibert_stack(hilbert_stack)
print(retiled_hilbert_image.size)
retiled_hilbert_image.show()
