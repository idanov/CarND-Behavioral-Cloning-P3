import argparse
import random
import numpy as np

from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.optimizers import Adam

from sdc.generator import generate_images_from
from sdc.load import load_csv, add_side_cam_images, strip_side_cam_images
from sdc.model import build_nvidia

###############################################
# Parse arguments
###############################################
parser = argparse.ArgumentParser(description='Steering angle model trainer')
parser.add_argument('--train', type=str, default="data/", help='Directory with the training data.')
parser.add_argument('--train2', type=str, default="", help='Directory with the training data.')
parser.add_argument('--train3', type=str, default="", help='Directory with the training data.')
parser.add_argument('--validation', type=str, default="validation/", help='Directory with the validation data.')
parser.add_argument('--output', type=str, default="model", help='Name of the output files.')
parser.add_argument('--best', type=str, default="", help='Name of the best model so far for transfer learning.')
parser.add_argument('--epochs', type=int, default=10, help='Number of epochs.')
parser.add_argument('--batch', type=int, default=128, help='Batch size.')
parser.add_argument('--n-rows', type=int, default=160, help='Number of rows in images.')
parser.add_argument('--n-cols', type=int, default=320, help='Number of columns in images.')
parser.add_argument('--n-ch', type=int, default=3, help='Number of channels in images.')
parser.add_argument('--side-cam', type=float, default=0.25, help='Side cam offset.')
parser.add_argument('--n-angle-bins', type=int, default=20, help='Number of bins for the angles histogram.')
args = parser.parse_args()

####################
# Training data
####################
data_train = load_csv(args.train, 'driving_log.csv')
if len(args.train2) > 0:
    more_data_train = load_csv(args.train2, 'driving_log.csv')
    data_train = data_train + more_data_train

if len(args.train3) > 0:
    more_data_train = load_csv(args.train3, 'driving_log.csv')
    data_train = data_train + more_data_train

# Get all the main cam images (used for visualising)
data_train_main_cam = strip_side_cam_images(data_train)
nb_train_main_cam_samples = len(data_train_main_cam)
# Add side cam images
data_train = add_side_cam_images(data_train, args.side_cam)
# Ensure all batches are full for convenience
nb_train_samples = (len(data_train) // args.batch) * args.batch
data_train = data_train[:nb_train_samples]
# Shuffle the data
random.shuffle(data_train)

#######################
# Validation data
#######################
data_validation = load_csv(args.validation, 'driving_log.csv')
data_validation = add_side_cam_images(data_validation, args.side_cam)
# Ensure all batches are full for convenience
nb_validation_samples = (len(data_validation) // (args.batch * 2)) * args.batch
data_validation = data_validation[:nb_validation_samples]

print("# training samples: ", nb_train_samples)
print("# training samples (main cam only): ", nb_train_main_cam_samples)
print("# validation samples: ", nb_validation_samples)

##########################################
# Dataset re-balancing
##########################################
y_train = np.array([a for a, _ in data_train])
y_validation = np.array([a for a, _ in data_validation])

train_hist, bin_edges = np.histogram(np.abs(y_train), bins=args.n_angle_bins, range=(0, 1), density=False)
uniform = nb_train_samples / args.n_angle_bins
weights = uniform / train_hist
print("Histogram of angles (training): ", train_hist)
print("Histogram bin edges: ", bin_edges)
print("Balancing weights: ", weights)

###################################
# Define the model
###################################
model = build_nvidia(args.n_rows, args.n_cols, args.n_ch)
model.compile(optimizer=Adam(lr=1e-4), loss="mse")
model.summary()

##################################################
# Train the model
##################################################
checkpoint = ModelCheckpoint(args.output + ".{epoch:02d}.h5", monitor='val_loss', verbose=1, save_best_only=False,
                             save_weights_only=False, mode='auto')
early_stopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=3, verbose=1, mode='auto')

history = model.fit_generator(
    generate_images_from(data_train, args.batch, random_changes=True),
    samples_per_epoch=nb_train_samples,
    nb_epoch=args.epochs,
    verbose=1,
    validation_data=generate_images_from(data_validation, args.batch, random_changes=False),
    nb_val_samples=nb_validation_samples,
    callbacks=[checkpoint, early_stopping]
)
