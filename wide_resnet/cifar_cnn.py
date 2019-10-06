import sys

import numpy as np
from sklearn.model_selection import train_test_split

import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Layer, BatchNormalization, AveragePooling2D, Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.regularizers import l2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint

if len(sys.argv) > 1:
    dataset = sys.argv[1]
else:
    dataset = "cifar10"

if dataset == "cifar100":
    print("cifar100: \n")
    from tensorflow.keras.datasets import cifar100 as cifar
else:
    print("cifar10: \n")
    from tensorflow.keras.datasets import cifar10 as cifar

RANDOM_SEED = 31415
BATCH_SIZE = 100
NUM_EPOCH = 50
L2_PENALTY = 1e-4

tf.random.set_seed(RANDOM_SEED)

class ConvExpansion(Layer):
    def __init__(self, filters, width=1):
        super(ConvExpansion, self).__init__()
        self.filters = filters
        self.width = width

    def call(self, block_input):
        out = Conv2D(self.filters*self.width, kernel_size=[3, 3], strides=[2, 2], padding="same",
            kernel_initializer = "he_normal",
            kernel_regularizer = keras.regularizers.l2(L2_PENALTY), use_bias=False)(block_input)
        
        out = BatchNormalization(momentum=0.1, epsilon=1e-5, gamma_initializer='uniform')(out)
        out = Activation("relu")(out)
        out = Conv2D(self.filters*self.width, kernel_size=[3, 3], strides=[1, 1], padding="same",
            kernel_initializer = "he_normal",
            kernel_regularizer = keras.regularizers.l2(L2_PENALTY), use_bias=False)(out)
        skip = Conv2D(self.filters*self.width, kernel_size=[1, 1], strides=[2, 2], padding="same",
            kernel_initializer = "he_normal",
            kernel_regularizer = keras.regularizers.l2(L2_PENALTY), use_bias=False)(block_input)
        out = keras.layers.add([out, skip])
        return out
        

class ResNetBlock(Layer):
    def __init__(self, filters, width=1, depth=1, dropout_ratio = 0.0):
        super(ResNetBlock, self).__init__()
        self.filters = filters
        self.depth = depth
        self.width = width
        self.dropout_ratio = dropout_ratio
        
    def call(self, block_input):
        residual = block_input
        out = block_input

        for i in range(self.depth):
            out = BatchNormalization(momentum=0.1, epsilon=1e-5, gamma_initializer='uniform')(out)
            out = Activation("relu")(out)
            out = Conv2D(filters=self.filters*self.width, kernel_size=[3, 3], strides = [1,1], padding="same",
                kernel_regularizer = keras.regularizers.l2(L2_PENALTY),
                kernel_initializer='he_normal',
                use_bias = False)(out)
            if self.dropout_ratio and i % 2 == 0:
                out = Dropout(self.dropout_ratio)(out)
        
        out = keras.layers.add([residual,out])
        out = BatchNormalization(momentum=0.1, epsilon=1e-5, gamma_initializer='uniform')(out)
        out = Activation("relu")(out)
        return out

class Model:
    def __init__(self, img_shape, num_labels, width=2,depth=16, dropout_ratio = 0.3):

        base_filter_numbers = img_shape[0]
        depth_per_resnet_block = (depth - 4)//6

        self.sequential_model = Sequential([
            Conv2D(base_filter_numbers, (3,3), activation="relu", input_shape = img_shape,
            kernel_initializer='he_normal'),
            BatchNormalization(momentum=0.1, epsilon=1e-5, gamma_initializer='uniform'),
            Activation('relu'),
            ConvExpansion(base_filter_numbers*2, width),
            ResNetBlock(base_filter_numbers*2, width, depth_per_resnet_block, dropout_ratio),
            ConvExpansion(base_filter_numbers*4, width),
            ResNetBlock(base_filter_numbers*4, width, depth_per_resnet_block, dropout_ratio),
            AveragePooling2D((8,8)),
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
            validation_data=(X_val, y_val),
            callbacks=[ModelCheckpoint("{}_model_weights.h5".format(dataset),
                                                monitor="val_acc",
                                                save_best_only=True,
                                                verbose=1)],
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
    data = cifar.load_data()
    (X_training_set, y_training_set), (X_test, y_test) = data
    X_training_set = normalize_images(X_training_set)
    X_test = normalize_images(X_test)

    img_shape = X_training_set.shape[1:]
    num_labels = np.amax(y_training_set[:]) + 1

    y_training_set, y_test = format_labels(y_training_set, num_labels), format_labels(y_test, num_labels), 
    X_train, X_val, y_train, y_val = train_test_split(X_training_set, y_training_set, test_size=1/6, random_state=RANDOM_SEED)
    
    datagen = ImageDataGenerator(
        rotation_range=10,
        width_shift_range=5./32,
        height_shift_range=5./32,
        horizontal_flip=True,
        zca_whitening=True
    )
    datagen.fit(X_train)

    model = Model(img_shape, num_labels)
    model_log = model.train(datagen, X_train, y_train, X_val, y_val)
    model.test(X_test, y_test)
