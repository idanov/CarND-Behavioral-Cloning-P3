import cv2
import numpy as np
import random
from sdc.processing import rotate_image, shift_image, scale_brightness, crop_image, resize_image, read_image

# Hardcoded params
shift_std = 15
rotation_std = 5
brightness_std = 0.25
adjust_brightness = True
random_flip = True
random_shift = False
random_rotation = False


def batch(iterable, n=1):
    l = len(iterable)
    for ndx in range(0, l, n):
        yield iterable[ndx:min(ndx + n, l)]


def perturb(img, angle):
    # Add random shifts and rotations
    if random_shift and random.randint(0, 1) == 1:
        img, angle = shift_image(img, angle, round(np.random.normal() * shift_std))
    if random_rotation and random.randint(0, 1) == 1:
        img, angle = rotate_image(img, angle, round(np.random.normal() * rotation_std))
    # Randomly flip the image
    if random_flip and random.randint(0, 1) == 1:
        img = cv2.flip(img, 1)
        angle = -angle
    # Add random brightness
    if adjust_brightness:
        img = scale_brightness(img, np.random.normal() * brightness_std)
    return [angle, img]


def preprocess(img, angle):
    img = crop_image(img, 35, 25)
    img = resize_image(img, 66, 200)
    return [angle, img]


def generate_images_from(data_rows, batch_size, random_changes=False, preprocessing=False, weights=None):
    while True:
        for b in batch(data_rows, batch_size):
            # Load images
            loaded_batch = [[angle, read_image(img_filename)] for angle, img_filename in b]
            # Add random changes
            if random_changes:
                loaded_batch = [perturb(img, angle) for angle, img in loaded_batch]
            # Add preprocessing
            if preprocessing:
                loaded_batch = [preprocess(img, angle) for angle, img in loaded_batch]

            # Generate a batch for angles
            angle_list = [angle for angle, _ in loaded_batch]
            angle_batch = np.transpose(np.array([angle_list], dtype=np.float32))
            # Generate a batch for images
            img_list = [np.expand_dims(img, axis=0) for _, img in loaded_batch]
            img_batch = np.vstack(img_list)

            weight_batch = np.ones_like(angle_batch[:, 0])
            # Add different weights to samples in according to their angle bucket
            if weights is not None:
                idx = np.abs(angle_batch[:, 0]) * len(weights)
                idx = idx.astype(np.int)
                weight_batch = weights[idx]
            yield (img_batch, angle_batch, weight_batch)


def generate_only_images_from(data_rows, batch_size, random_changes=False):
    for img, _, _ in generate_images_from(data_rows, batch_size, random_changes=random_changes):
        yield img
