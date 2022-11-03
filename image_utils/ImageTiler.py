from PIL import Image
import numpy as np
from hilbertcurve.hilbertcurve import HilbertCurve

def scale_rgb(arr):
    max_a = max(arr)
    min_a = min(arr)
    return np.round(255*(arr - min_a)/(max_a - min_a + 1e-5))

def add_margin(pil_img, top, right, bottom, left, color):
    width, height = pil_img.size
    new_width = width + right + left
    new_height = height + top + bottom
    result = Image.new(pil_img.mode, (new_width, new_height), color)
    result.paste(pil_img, (left, top))
    return result

def get_tiled_image(img: Image, tile_height: int, tile_width: int):
    W,H = img.size
    print(f"H/W: {H}/{W}")
    n_rows = H//tile_height
    n_cols = W//tile_width
    img = np.array(img)
    tiles = np.zeros(shape=(n_rows, n_cols, tile_height, tile_width, 3),dtype=np.uint8)
    for row in range(n_rows):
        for col in range(n_cols):
            this_row, next_row = tile_height*row, tile_height*(row+1)
            this_col, next_col = tile_width*col, tile_width*(col+1)
            tile = img[this_row:next_row, this_col:next_col].astype(np.uint8)
            try:
                tiles[row,col] = tile
            except ValueError as e:
                print(f"row/col = {row}/{col}")
                print(f"row*th/col*tw = {row*tile_height}/{col*tile_width}")
                raise e
    return tiles

def get_hilbert_tile_stack(img: Image, tile_height: int, tile_width: int):
    tiles = get_tiled_image(img, tile_height, tile_width)
    n_rows, n_cols, _, _, _ = tiles.shape 

    side_length = max(n_rows, n_cols)
    hilbert_order = int(np.ceil(np.log2(side_length)))
    
    points = [(n,m) for n in range(n_rows) for m in range(n_cols)]
    positions = HilbertCurve(hilbert_order, 2).distances_from_points(points)
    tile_stack = np.zeros(shape=(max(positions)+1, tile_height, tile_width, 3), dtype=np.uint8)
    for point,pos in zip(points,positions):
        tile_stack[pos] = tiles[point[0],point[1]]
    return tile_stack

def retile_hibert_stack(hilbert_tiles: np.array):
    T, th, tw, tc = hilbert_tiles.shape

    hilbert_dim = int(np.ceil(np.log2(T)/2))
    square_size = int(2**hilbert_dim)
    H,W = square_size*th, square_size*tw
    new_img  = np.zeros(shape=(H, W, tc), dtype=np.uint8)
    
    new_width = W
    new_height = H
    positions = list(range(T))
    points = HilbertCurve(hilbert_dim, 2).points_from_distances(positions)
    for point,pos in zip(points,positions):
        row = point[0]
        col = point[1]
        this_row, next_row = th*row, th*(row+1)
        this_col, next_col = tw*col, tw*(col+1)
        tile = hilbert_tiles[pos]
        new_img[this_row:next_row, this_col: next_col] = tile
        if not tile.any():
            if row == 0 and col*tw < new_width:
                new_width = col*tw
            if col == 0 and row*th < new_height:
                new_height = row*th
    return Image.fromarray(new_img[:new_height,:new_width,:], 'RGB')

def retile_image(tiles: np.array, show_only=None):
    N,M,th,tw,C = tiles.shape
    H,W = th*N, tw*M
    output_img = np.zeros((H, W, C),dtype=np.uint8)
    for row in range(N):
        for col in range(M):
            this_row, next_row = tw*row, tw*(row+1)
            this_col, next_col = th*col, th*(col+1)
            if show_only is not None:
                if (row,col) in show_only:
                    output_img[this_row:next_row, this_col:next_col] = tiles[row,col]
            else:
                output_img[this_row:next_row, this_col:next_col] = tiles[row,col]
    return Image.fromarray(output_img,'RGB')

