import sys
import numpy as np
import sklearn.metrics as metrics
from sklearn.model_selection import train_test_split
import tensorflow.keras as keras
import tensorflow.keras.callbacks as callbacks
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import backend as K
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Layer, Input, Add, Activation, Dropout, Flatten, Dense, Conv2D, MaxPooling2D, AveragePooling2D, BatchNormalization
from tensorflow.keras.regularizers import l2

L2_PENALTY = 0.0005
CHANNEL_AXIS = 1 if K.image_data_format() == "channels_first" else -1
RANDOM_SEED = 31415
BATCH_SIZE = 100
NUM_EPOCHS = 50
DROPOUT_RATE = 0.00


class InitialConv(Layer):
    def __init__(self, img_shape):
        super(InitialConv, self).__init__()
        self.img_shape = img_shape
        self.conv_layer = Conv2D(16, (3, 3), padding='same', kernel_initializer='he_normal',
                        kernel_regularizer=l2(L2_PENALTY),
                        use_bias=False, input_shape = self.img_shape)
        self.bn_layer = BatchNormalization(axis=CHANNEL_AXIS, momentum=0.1, epsilon=1e-5, gamma_initializer='uniform')
        self.activation_layer = Activation('relu')
    def call(self, input_layer):
        x = self.conv_layer(input_layer)
        x = self.bn_layer(x)
        x = self.activation_layer(x)
        return x

class ExpandConv(Layer):
    def __init__(self, base, k, strides = (1,1)):
        super(ExpandConv, self).__init__()
        self.base = base
        self.k = k
        self.strides = strides

        self.conv_1 = Conv2D(self.base * self.k, (3, 3), padding='same', strides = self.strides, kernel_initializer='he_normal',
                        kernel_regularizer=l2(L2_PENALTY),
                        use_bias=False)
        
        self.bn = BatchNormalization(axis=CHANNEL_AXIS, momentum=0.1, epsilon=1e-5, gamma_initializer='uniform')

        self.activation = Activation('relu')

        self.conv_2 = Conv2D(self.base * self.k, (3, 3), padding='same', kernel_initializer='he_normal',
                        kernel_regularizer=l2(L2_PENALTY),
                        use_bias=False)

        self.skip_conv = Conv2D(self.base * self.k, (1, 1), padding='same', strides = self.strides, kernel_initializer='he_normal',
                        kernel_regularizer=l2(L2_PENALTY),
                        use_bias=False)

        self.add = Add()

    def call(self, input_layer):
        x = self.conv_1(input_layer)

        x = self.bn(x)
        x = self.activation(x)

        x = self.conv_2(x)

        skip = self.skip_conv(input_layer)

        m = self.add([x, skip])

        return m

# num_filters: 16, then 32, then 64

class ConvBlock(Layer):
    def __init__(self, num_filters, k=1, dropout=0.0):
        super(ConvBlock, self).__init__()
        self.num_filters = num_filters
        self.k = k
        self.dropout = dropout

        self.bn_layer_1 = BatchNormalization(axis=CHANNEL_AXIS, momentum=0.1, epsilon=1e-5, gamma_initializer='uniform')
        self.activation_1 = Activation('relu')
        self.conv_1 = Conv2D(self.num_filters * self.k, (3, 3), padding='same', kernel_initializer='he_normal',
                        kernel_regularizer=l2(L2_PENALTY),
                        use_bias=False)
        self.dropout_layer = Dropout(self.dropout)
        self.bn_layer_2 = BatchNormalization(axis=CHANNEL_AXIS, momentum=0.1, epsilon=1e-5, gamma_initializer='uniform')
        self.activation_2 = Activation('relu')
        self.conv_2 = Conv2D(self.num_filters * self.k, (3, 3), padding='same', kernel_initializer='he_normal',
                        kernel_regularizer=l2(L2_PENALTY),
                        use_bias=False)

        self.add = Add()

    def call(self, input_layer):

        init = input_layer

        x = self.bn_layer_1(input_layer)
        x = self.activation_1(x)
        x = self.conv_1(x)

        if self.dropout > 0.0:
            x = self.dropout_layer(x)

        x = self.bn_layer_2(x)
        x = self.activation_2(x)
        x = self.conv_2(x)

        m = self.add([init, x])
        return m

class BatchActivate(Layer):
    def __init__(self):
        super(BatchActivate, self).__init__()
    
    def call(self, input_layer):
        x = input_layer
        x = BatchNormalization(axis=CHANNEL_AXIS, momentum=0.1, epsilon=1e-5, gamma_initializer='uniform')(x)
        x = Activation('relu')(x)
        return x

class WideResNet:
    def __init__(self, img_shape, num_labels=10, N=2, k=1, dropout=0.0):
        self.img_shape = img_shape
        self.num_labels = num_labels
        self.N = N
        self.k = k
        self.dropout = dropout

        self.input_layer = Input(shape = img_shape)
        self.initial_convolution = InitialConv(img_shape)
        self.expansion_1 = ExpandConv(16, k, strides = (1,1))
        self.conv_1 = ConvBlock(16, k, dropout)
        self.batchactive_1 = BatchActivate()
        self.expansion_2 = ExpandConv(32, k, strides = (2,2))
        self.conv_2 = ConvBlock(32, k, dropout)
        self.batchactive_2 = BatchActivate()
        self.expansion_3 = ExpandConv(64, k, strides = (2,2))
        self.conv_3 = ConvBlock(64, k, dropout)
        self.batchactive_3 = BatchActivate()
        self.avgpool = AveragePooling2D((8,8))
        self.flatten = Flatten()
        self.dense = Dense(num_labels, kernel_regularizer=l2(L2_PENALTY), activation='softmax')

        self.model = Sequential([self.input_layer, self.initial_convolution, self.expansion_1,
            self.conv_1, self.batchactive_1, self.expansion_2,
            self.conv_2, self.batchactive_2, self.expansion_3,
            self.conv_3, self.batchactive_3,
            self.avgpool, self.flatten, self.dense])

    def compile(self):

        self.model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["acc", "top_k_categorical_accuracy"])

        print("Finished compiling")
        self.model.summary()
    
    def fit_generator(self, generator, X_train, y_train, X_val, y_val):
        self.model_log = self.model.fit_generator(
            generator.flow(X_train, y_train, batch_size=BATCH_SIZE),
            steps_per_epoch=len(X_train) // BATCH_SIZE,
            epochs=NUM_EPOCHS,
            callbacks=[callbacks.ModelCheckpoint("WRN-16-8 Weights.h5",
                monitor="val_acc",
                save_best_only=True,
                verbose=1)],
            validation_data=(X_val, y_val),
            validation_steps=X_val.shape[0] // BATCH_SIZE)

        return self.model_log


    def test(self, X_test, y_test):
        self.model.evaluate(X_test, y_test, verbose = 1)
        

def get_dataset():
    (X_train, y_train), (X_test, y_test) = cifar.load_data()

    X_train = X_train.astype('float32')
    X_train = (X_train - X_train.mean(axis=0)) / (X_train.std(axis=0))

    X_test = X_test.astype('float32')
    X_test = (X_test - X_test.mean(axis=0)) / (X_test.std(axis=0))

    y_train = keras.utils.to_categorical(y_train)
    y_test = keras.utils.to_categorical(y_test)

    
    return X_train, y_train, X_test, y_test

def split_training_set(X_train, y_train, test_size=1/10):
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=test_size, random_state=RANDOM_SEED)
    return X_train, y_train, X_val, y_val

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


X_train, y_train, X_test, y_test = get_dataset()
#X_train, X_val, y_train, y_val = split_training_set(X_train, y_train)

generator = ImageDataGenerator(rotation_range=10,
                               width_shift_range=5./32,
                               height_shift_range=5./32,)

img_shape = X_train.shape[1:]
num_labels = len(y_train[0])

# For WRN-16-8 put N = 2, k = 8
# For WRN-28-10 put N = 4, k = 10
# For WRN-40-4 put N = 6, k = 4

model = WideResNet(img_shape, num_labels=num_labels, N=2, k=8, dropout=DROPOUT_RATE)
model.compile()

#model.load_weights("weights/WRN-16-8 Weights.h5")
#print("Model loaded.")

model.fit_generator(generator, X_train, y_train, X_test, y_test)

model.test(X_test, y_test)
#yPreds = model.predict(X_test)
# yPred = np.argmax(yPreds, axis=1)
# yPred = keras.utils.to_categorical(yPred)
# yTrue = y_test

#accuracy = metrics.accuracy_score(yTrue, yPred) * 100
#error = 100 - accuracy
#print("Accuracy : ", accuracy)
#print("Error : ", error)

