from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Lambda, ELU
from keras.layers.convolutional import Convolution2D, Cropping2D


# Function used by the resize layer
def resize(img):
    import tensorflow as tf
    n_rows_after, n_cols_after, n_ch_after = 66, 200, 3
    return tf.image.resize_images(img, (n_rows_after, n_cols_after))


# A function building an Nvidia CNN model with some hardcoded dropout layers
def build_nvidia(n_rows, n_cols, n_ch):
    model = Sequential()
    model.add(Cropping2D(cropping=((55, 25), (0, 0)), input_shape=(n_rows, n_cols, n_ch)))
    model.add(Lambda(resize))
    model.add(Lambda(lambda x: (x / 255.0) - 0.5))
    model.add(Convolution2D(24, 5, 5, subsample=(2, 2), border_mode='valid'))
    model.add(ELU())
    model.add(Convolution2D(36, 5, 5, subsample=(2, 2), border_mode='valid'))
    model.add(ELU())
    model.add(Convolution2D(48, 5, 5, subsample=(2, 2), border_mode='valid'))
    model.add(Dropout(.5))
    model.add(ELU())
    model.add(Convolution2D(64, 3, 3, border_mode='valid'))
    model.add(Dropout(.5))
    model.add(ELU())
    model.add(Convolution2D(64, 3, 3, border_mode='valid'))
    model.add(Dropout(.7))
    model.add(ELU())
    model.add(Flatten())
    model.add(Dense(100))
    model.add(Dropout(.5))
    model.add(ELU())
    model.add(Dense(50))
    model.add(Dropout(.2))
    model.add(ELU())
    model.add(Dense(10))
    model.add(ELU())
    model.add(Dense(1, activation="tanh"))
    return model
