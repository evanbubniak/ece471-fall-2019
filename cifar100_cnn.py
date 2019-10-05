import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

import numpy as np
from sklearn.model_selection import train_test_split

import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.datasets import cifar100
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Layer, BatchNormalization, AveragePooling2D, Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.regularizers import l2
from tensorflow.keras.preprocessing.image import ImageDataGenerator

RANDOM_SEED = 31415
BATCH_SIZE = 100
NUM_EPOCH = 50
DROPOUT_RATIO = 0.1
L2_PENALTY = 0.001
WEIGHT_DECAY = 1e-4

tf.random.set_seed(RANDOM_SEED)

class Model:
    def __init__(self, img_dim, num_labels):

        self.sequential_model = Sequential([
            Conv2D(img_dim, (3,3), activation="relu", input_shape = (img_dim, img_dim, 3)),
            MaxPooling2D(pool_size = (2,2)),
            Dropout(DROPOUT_RATIO),
            Conv2D(img_dim*2, (3,3), activation="relu"),
            MaxPooling2D(pool_size = (2,2)),
            Dropout(DROPOUT_RATIO),
            Conv2D(img_dim*4, (3,3), activation="relu"),
            MaxPooling2D(pool_size = (2,2)),
            Dropout(DROPOUT_RATIO),
            Flatten(),
            Dense(num_labels, activation = "softmax", kernel_regularizer = keras.regularizers.l2(L2_PENALTY))
            ])
        
        self.sequential_model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adam(),
              metrics=['accuracy', 'top_k_categorical_accuracy'])
    
    def train(self, datagen, X_train, y_train, X_val, y_val):
        self.model_log = self.sequential_model.fit_generator(
            datagen.flow(X_train, y_train, batch_size = BATCH_SIZE),
            steps_per_epoch = X_train.shape[0] // BATCH_SIZE,           
            epochs=NUM_EPOCH,
            verbose=1,
            validation_data=(X_val, y_val)
        )
        return self.model_log
    
    def test(self, X_test, y_test):
        self.sequential_model.evaluate(X_test, y_test, verbose = 1)
        self.sequential_model.summary()

def normalize_images(x):
    return x.astype("float32") / 255

def format_labels(label_set, num_labels):
    return keras.utils.to_categorical(label_set, num_labels)

if __name__ == "__main__":
    data = cifar100.load_data()
    (X_training_set, y_training_set), (X_test, y_test) = data
    X_training_set = normalize_images(X_training_set)
    X_test = normalize_images(X_test)

    img_dim = X_training_set.shape[1]
    num_labels = np.amax(y_training_set[:]) + 1

    y_training_set, y_test = format_labels(y_training_set, num_labels), format_labels(y_test, num_labels), 
    X_train, X_val, y_train, y_val = train_test_split(X_training_set, y_training_set, test_size=1/6, random_state=RANDOM_SEED)
    
    datagen = ImageDataGenerator(
        rotation_range=15,
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=True,
    )
    datagen.fit(X_train)
    
    model = Model(img_dim, num_labels)
    model_log = model.train(datagen, X_train, y_train, X_val, y_val)
    model.test(X_test, y_test)

