from PIL import Image, ImageChops, ImageFile
import numpy as np
from pathlib import Path
from tqdm import tqdm
import os
from joblib import Parallel, delayed


THRESHOLD = 20
IMAGE_PATH = Path("../dataset")
TRAIN_PATH = IMAGE_PATH / "images_train"
TEST_PATH = IMAGE_PATH / "images_test"
TRAIN_NEW_PATH = IMAGE_PATH / "images_train_unpadded"
TEST_NEW_PATH = IMAGE_PATH / "images_test_unpadded"


def remove_padding(image):
    image = image.convert('RGB')
    background_color = image.getpixel((0,0))
    background = Image.new(image.mode, image.size, background_color)
    difference = ImageChops.difference(image, background)
    difference = ImageChops.add(difference, difference, 2.0, -THRESHOLD)
    bbox = difference.getbbox()
    if bbox:
        return image.crop(bbox)
    return image


def process_image(filename, start_path, end_path):
    ImageFile.LOAD_TRUNCATED_IMAGES = True
    try:
        image = Image.open(start_path / filename)
        image = remove_padding(image)
        image.save(end_path / filename)
    except OSError:
        pass


if __name__ == '__main__':
    path_pairs = [(TRAIN_PATH, TRAIN_NEW_PATH), (TEST_PATH, TEST_NEW_PATH)]

    for from_folder, to_folder in path_pairs:
        to_folder.mkdir(parents=True, exist_ok=True)
        files = os.listdir(from_folder)
        Parallel(n_jobs=os.cpu_count())(delayed(process_image)(filename, from_folder, to_folder) for filename in tqdm(files))
        print(f"Folder {str(TRAIN_PATH)} processed")
