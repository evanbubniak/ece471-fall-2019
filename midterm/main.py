import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-m', '--model_num',
    nargs = '*', default = [1, 2, 3, 4, 5],
    help = "model number; specify 1, 2, 3, 4, 5, or some list thereof.")
parser.add_argument('-d', '--data_corruption_types',
    nargs = '*', default = ["true_labels", "random_labels", 
                   "shuffled_pixels", "random_pixels", "gaussian"],
    help = "Data corruption type; select true_labels, random_labels, shuffled_pixels, random_pixels, gaussian or some list thereof.")
args = parser.parse_args()

BATCH_SIZE = 125
NUM_SAMPLES = 50000
NUM_EPOCHS = 100
STEPS_PER_EPOCH = ceil(NUM_SAMPLES / BATCH_SIZE)


import tensorflow.keras as keras
from tensorflow.keras.datasets import cifar10
from utils import *
from math import ceil
import sys

def get_model(model_code):
    if model_code == 1:
        return MiniInceptionV3(
            input_shape = X.shape[1:],
            num_labels = 10
            )
    elif model_code == 2:
        return MiniInceptionV3(
            input_shape = X.shape[1:],
            num_labels = 10,
            use_batch_norm = False
            )
    elif model_code == 3:
        return AlexNet(
            input_shape = X.shape[1:],
            num_labels = 10
        )
    elif model_code == 4:
        return MLP(
            input_shape = X.shape[1:],
            num_labels = 10,
            num_hidden_layers = 1
        )
    elif model_code == 5:
        return MLP(
            input_shape = X.shape[1:],
            num_labels = 10,
            num_hidden_layers = 3
        )
        
(X_train, y_train), (X_test, y_test) = cifar10.load_data()

X_train = preprocess_input(X_train)
X_test = preprocess_input(X_test)
y_train = preprocess_labels(y_train)
y_test = preprocess_labels(y_test)

inputs = {
    "true_labels": [X_train, y_train, X_test, y_test],
    "random_labels": [X_train, randomize_labels(y_train.shape[0], 10), 
                 X_test, randomize_labels(y_test.shape[0], 10)],
    "shuffled_pixels": [shuffle_pixels(X_train), y_train, 
                   shuffle_pixels(X_test), y_test],
    "random_pixels": [randomize_pixels(X_train), y_train, 
                 randomize_pixels(X_test), y_test],
    "gaussian": [create_gaussian_noise(X_train), y_train, 
            create_gaussian_noise(X_test), y_test]}

for model_code in args.model_num:
    for corruption_type in args.data_corruption_types:
        test_X = inputs[corruption_type][2]
        test_y = inputs[corruption_type][3]
        model = get_model(model_code)
        model.compile()
        model.fit(*inputs[corruption_type],
            NUM_EPOCHS, corruption_type, BATCH_SIZE)
        model.evaluate(test_X, test_y)
