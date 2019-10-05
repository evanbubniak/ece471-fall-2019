import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

import numpy as np
from sklearn.model_selection import train_test_split

import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Layer, BatchNormalization, AveragePooling2D, Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D

RANDOM_SEED = 31415
BATCH_SIZE = 100
NUM_EPOCH = 10
DROPOUT_RATIO = 0.5
L2_PENALTY = 0.001

tf.random.set_seed(RANDOM_SEED)

class ResNetUnit(Layer):
    def __init__(self, filters, pool=False):
        super(ResNetUnit, self).__init__()
        self.filters = filters
        self.pool = pool
        
    def call(self, x):
        res = x
        print(x.shape)
        if self.pool:
            x = MaxPooling2D(pool_size=(2, 2), padding = "same")(x)
            res = Conv2D(filters=self.filters,kernel_size=[1,1],strides=(2,2),padding="same")(res)
        out = BatchNormalization()(x)
        out = Activation("relu")(out)
        out = Conv2D(filters=self.filters, kernel_size=[3, 3], strides=[1, 1], padding="same")(out)

        print(out.shape)
        
        out = BatchNormalization()(out)
        out = Activation("relu")(out)
        out = Conv2D(filters=self.filters, kernel_size=[3, 3], strides=[1, 1], padding="same")(out)
        out = keras.layers.add([res,out])

        return out

class Model:
    def __init__(self, img_dim, num_labels):

        self.sequential_model = Sequential([
            Conv2D(filters=img_dim, kernel_size=[3, 3], strides=[1, 1], input_shape = (img_dim, img_dim, 3)),
            ResNetUnit(img_dim),
            ResNetUnit(img_dim*2, pool=True),
            ResNetUnit(img_dim*4, pool=True),
            BatchNormalization(),
            Activation("relu"),
            AveragePooling2D(pool_size=(8,8), strides = (1,1), padding="same"),
            Flatten(),
            Dense(num_labels, activation='softmax')
            ])
        
        self.sequential_model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.RMSprop(),
              metrics=['accuracy', 'top_k_categorical_accuracy'])
    
    def train(self, X_train, y_train, X_val, y_val):
        self.model_log = self.sequential_model.fit(X_train, y_train,
            batch_size=BATCH_SIZE,
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
    data = cifar10.load_data()
    (X_training_set, y_training_set), (X_test, y_test) = data
    X_training_set = normalize_images(X_training_set)
    X_test = normalize_images(X_test)

    img_dim = X_training_set.shape[1]
    num_labels = np.amax(y_training_set[:]) + 1

    y_training_set, y_test = format_labels(y_training_set, num_labels), format_labels(y_test, num_labels), 
    X_train, X_val, y_train, y_val = train_test_split(X_training_set, y_training_set, test_size=1/6, random_state=RANDOM_SEED)

    model = Model(img_dim, num_labels)
    model_log = model.train(X_train, y_train, X_val, y_val)
    model.test(X_test, y_test)
