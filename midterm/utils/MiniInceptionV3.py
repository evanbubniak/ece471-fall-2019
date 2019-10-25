import tensorflow.keras as keras
import os


INITIAL_LEARNING_RATE = 0.1
RATE_DECAY_FACTOR_PER_EPOCH = 0.95
MOMENTUM_PARAMETER = 0.9
BATCH_SIZE = 128

def learning_rate_schedule(epoch_num):
    return INITIAL_LEARNING_RATE*(RATE_DECAY_FACTOR_PER_EPOCH)**(epoch_num)

if keras.backend.image_data_format() == 'channels_first':
    CHANNEL_AXIS = 1
else:
    CHANNEL_AXIS = 3

if keras.backend.image_data_format() == 'channels_first':
    BATCHNORM_AXIS = 1
else:
    BATCHNORM_AXIS = 3

def conv_module(x,
              filters,
              kernel_size,
              padding='same',
              strides=(1, 1)):
    """Utility function to apply conv + BN.
    # Arguments
        x: input tensor.
        filters: filters in `Conv2D`.
        num_row: height of the convolution kernel.
        num_col: width of the convolution kernel.
        padding: padding mode in `Conv2D`.
        strides: strides in `Conv2D`.
        name: name of the ops; will become `name + '_conv'`
            for the convolution and `name + '_bn'` for the
            batch norm layer.
    # Returns
        Output tensor after applying `Conv2D` and `BatchNormalization`.
    """
    x = keras.layers.Conv2D(
        filters,
        kernel_size = kernel_size,
        strides=strides,
        padding=padding,
        use_bias=False)(x)
    x = keras.layers.BatchNormalization(axis=BATCHNORM_AXIS, scale=False)(x)
    x = keras.layers.Activation('relu')(x)
    return x

def inception_module(x, filters_1, filters_3):
    conv_module1 = conv_module(x, filters = filters_1, kernel_size = (1, 1), strides = (1, 1))
    conv_module3 = conv_module(x, filters = filters_3, kernel_size = (3, 3), strides = (1, 1))
    
    return keras.layers.concatenate(
        [conv_module1,
        conv_module3],
        axis = CHANNEL_AXIS)



def downsample_module(x, filters):
    max_pooling = keras.layers.MaxPooling2D(pool_size = (3,3), strides = (2,2), padding = 'same')(x)
    conv_module3 = conv_module(x, filters, kernel_size = (3, 3), strides = (2,2))

    return keras.layers.concatenate(
        [conv_module3,
        max_pooling],
        axis = CHANNEL_AXIS)

class MiniInceptionV3:
    def __init__(self, input_shape, num_labels=10):
        input_layer = keras.layers.Input(shape = input_shape)
        x = conv_module(input_layer, 96, kernel_size = (3,3), strides = (1,1))
        x = inception_module(x, 32, 32)
        x = inception_module(x, 32, 48)
        x = downsample_module(x, 80)
        x = inception_module(x, 112, 48)
        x = inception_module(x, 96, 64)
        x = inception_module(x, 80, 80)
        x = inception_module(x, 48, 96)
        x = downsample_module(x, 96)
        x = inception_module(x, 176, 160)
        x = inception_module(x, 176, 160)
        x = keras.layers.GlobalAveragePooling2D(data_format = keras.backend.image_data_format())(x)
        x = keras.layers.Dense(num_labels, activation='softmax', name='predictions')(x)

        self.model = keras.models.Model(input_layer, x, name='inception_v3')
    
    def compile(self):

        sgd = keras.optimizers.SGD(learning_rate = INITIAL_LEARNING_RATE,
            momentum = MOMENTUM_PARAMETER,
            nesterov = False
            )

        self.model.compile(loss="categorical_crossentropy", optimizer=sgd, metrics=["acc", "top_k_categorical_accuracy"])

        print("Finished compiling")
        self.model.summary()
    
    def fit(self, X_train, y_train, X_val, y_val, num_epoch, name, batch_size, load_weights = False):
        if load_weights:
            self.model.load_weights("{}-weights.h5".format(name))

        self.model_log = self.model.fit(X_train, y_train,
            batch_size = batch_size,
            epochs = num_epoch,
            verbose = 1,
            validation_data = (X_val, y_val),
            callbacks = [
                keras.callbacks.LearningRateScheduler(learning_rate_schedule, verbose = 0),
                keras.callbacks.ModelCheckpoint("{}-weights.h5".format(name),
                    monitor="val_acc",
                    save_best_only=True,
                    verbose=1),
                keras.callbacks.CSVLogger("{}.csv".format(name))]
                )

        return self.model_log

    def evaluate(self, X_test, y_test):
        self.model.evaluate(X_test, y_test, verbose = 1)


if __name__ == "__main__":
    
    model = MiniInceptionV3(
        input_shape = (28, 28, 3),
        num_labels = 10
    )
    model.compile()

