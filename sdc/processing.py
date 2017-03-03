import cv2
import numpy as np


def within(x, x_min, x_max):
    return max(min(x, x_max), x_min)


def read_image(filename):
    img = cv2.imread(filename)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def crop_image(img, y_top, y_bottom):
    return img[y_top:-y_bottom, :, :]


def resize_image(img, new_rows, new_cols):
    return cv2.resize(img, (new_cols, new_rows), interpolation=cv2.INTER_AREA)


def scale_brightness(img, ratio):
    img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    img[:, :, 2] = np.minimum(np.floor(img[:, :, 2] * ratio), 255)
    img = cv2.cvtColor(img, cv2.COLOR_HSV2RGB)
    return img


def shift_image(img, angle, x_shift, steering_per_pixel=0.02):
    rows, cols, _ = img.shape
    angle = within(angle + x_shift * steering_per_pixel, -1., 1.)
    mat = np.array([[1, 0, x_shift],
                    [0, 1, 0]], dtype=np.float32)
    img = cv2.warpAffine(img, mat, (cols, rows))
    return [angle, img]


def rotate_image(img, angle, degrees):
    rows, cols, _ = img.shape
    angle = within(angle - (degrees / 25), -1., 1.)
    mat = cv2.getRotationMatrix2D((cols / 2, rows), degrees, 1)
    img = cv2.warpAffine(img, mat, (cols, rows))
    return [angle, img]
