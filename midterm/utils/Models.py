import tensorflow.keras as keras
from tensorflow.keras.layers import Input, Conv2D, Activation, MaxPooling2D, Dense, Dropout, Flatten

RATE_DECAY_FACTOR_PER_EPOCH = 0.95
MOMENTUM_PARAMETER = 0.9
DROPOUT_RATE = 0.5

if keras.backend.image_data_format() == 'channels_first':
    CHANNEL_AXIS = 1
else:
    CHANNEL_AXIS = 3

if keras.backend.image_data_format() == 'channels_first':
    BATCHNORM_AXIS = 1
else:
    BATCHNORM_AXIS = 3

class MidtermModel:
    def __init__(self, weight_decay = None):
        self.model_name = "model"
        self.model = keras.models.Model(None, None)
        self.initial_learning_rate = 0.1
        self.weight_decay = weight_decay

    def compile(self):

        sgd = keras.optimizers.SGD(learning_rate = self.initial_learning_rate,
            momentum = MOMENTUM_PARAMETER,
            nesterov = False
            )

        self.model.compile(loss="categorical_crossentropy", optimizer=sgd, metrics=["acc", "top_k_categorical_accuracy"])

        print("Finished compiling")
        self.model.summary()
    
    def fit(self, X_train, y_train, X_val, y_val, num_epoch, data_name, batch_size, load_weights = False):
        run_name = "{}-{}".format(self.model_name, data_name)
        
        callbacks = [
                keras.callbacks.LearningRateScheduler(
                    self.learning_rate_schedule,
                    verbose = 0),
                keras.callbacks.ModelCheckpoint("{}-{}-weights.h5".format(data_name),
                    monitor="acc",
                    save_best_only=True,
                    verbose=1),
                keras.callbacks.CSVLogger(
                    "{}.csv".format(run_name))
                ]

        if load_weights:
            self.model.load_weights("{}-weights.h5".format(run_name))

        self.model_log = self.model.fit(X_train, y_train,
            batch_size = batch_size,
            epochs = num_epoch,
            verbose = 1,
            validation_data = (X_val, y_val),
            callbacks = callbacks)

        return self.model_log

    def evaluate(self, X_test, y_test):
        self.model.evaluate(X_test, y_test, verbose = 1)

    def learning_rate_schedule(self, epoch_num):
        return self.initial_learning_rate*(RATE_DECAY_FACTOR_PER_EPOCH)**(epoch_num)

class MiniInceptionV3(MidtermModel):
    def __init__(self, input_shape, num_labels=10):
        super(MiniInceptionV3, self).__init__()
        self.initial_learning_rate = 0.1
        self.model_name = "MiniInceptionV3"
        input_layer = keras.layers.Input(shape = input_shape)
        x = self.conv_module(input_layer, 96, kernel_size = (3,3), strides = (1,1))
        x = self.inception_module(x, 32, 32)
        x = self.inception_module(x, 32, 48)
        x = self.downsample_module(x, 80)
        x = self.inception_module(x, 112, 48)
        x = self.inception_module(x, 96, 64)
        x = self.inception_module(x, 80, 80)
        x = self.inception_module(x, 48, 96)
        x = self.downsample_module(x, 96)
        x = self.inception_module(x, 176, 160)
        x = self.inception_module(x, 176, 160)
        x = keras.layers.GlobalAveragePooling2D(data_format = keras.backend.image_data_format())(x)
        x = keras.layers.Dense(num_labels, activation='softmax', name='predictions')(x)

        self.model = keras.models.Model(input_layer, x, name='inception_v3')

    def conv_module(self, x, filters, kernel_size, padding='same', strides=(1, 1)):
        x = keras.layers.Conv2D(
            filters,
            kernel_size = kernel_size,
            strides=strides,
            padding=padding,
            use_bias=False)(x)
        x = keras.layers.BatchNormalization(axis=BATCHNORM_AXIS, scale=False)(x)
        x = keras.layers.Activation('relu')(x)
        return x

    def inception_module(self, x, filters_1, filters_3):
        conv_module1 = self.conv_module(x, filters = filters_1, kernel_size = (1, 1), strides = (1, 1))
        conv_module3 = self.conv_module(x, filters = filters_3, kernel_size = (3, 3), strides = (1, 1))
        
        return keras.layers.concatenate(
            [conv_module1,
            conv_module3],
            axis = CHANNEL_AXIS)

    def downsample_module(self, x, filters):
        max_pooling = keras.layers.MaxPooling2D(pool_size = (3,3), strides = (2,2), padding = 'same')(x)
        conv_module3 = self.conv_module(x, filters, kernel_size = (3, 3), strides = (2,2))

        return keras.layers.concatenate(
            [conv_module3,
            max_pooling],
            axis = CHANNEL_AXIS)

class AlexNet(MidtermModel):
    def __init__(self, input_shape, num_labels=10):
        super(AlexNet, self).__init__()
        self.initial_learning_rate = 0.01
        self.model_name = "AlexNet"
        input_layer = Input(shape = input_shape)
        x = Conv2D(96, kernel_size = (11, 11), strides = (4,4), padding = 'valid')(input_layer)
        x = Activation('relu')(x)
        x = MaxPooling2D(pool_size = (3,3), strides = (2,2), padding = "valid")(x)
        x = Conv2D(256, (5, 5), strides = (1,1), padding = 'valid')(x)
        x = Activation('relu')(x)
        x = MaxPooling2D(pool_size = (3,3), strides = (2,2), padding = "valid")
        x = Conv2D(384, (3, 3), strides = (1,1), padding = 'same')(x)
        x = Activation('relu')(x)
        x = Conv2D(384, (3, 3), strides = (1,1), padding = 'same')(x)
        x = Activation('relu')(x)
        x = Conv2D(256, (3, 3), strides = (1,1), padding = 'same')(x)
        x = Activation('relu')(x)
        x = MaxPooling2D(pool_size = (3,3), strides = (2,2))

        x = Flatten()(x)
        x = Dense(4096)(x)
        x = Activation('relu')(x)
        x = Dropout(DROPOUT_RATE)(x)
        x = Dense(4096)(x)
        x = Activation('relu')(x)
        x = Dropout(DROPOUT_RATE)(x)
        x = Dense(num_labels, activation='softmax', name='predictions')(x)

        self.model = keras.models.Model(input_layer, x, name=self.model_name)

class MLP(MidtermModel):
    def __init__(self, input_shape, num_labels=10, num_hidden_layers = 1, num_hidden_units = 512):
        super(MLP, self).__init__()
        self.model_name = "MLP {}x{}".format(num_hidden_layers, num_hidden_units)
        input_layer = Input(shape = input_shape)
        x = input_layer
        for hidden_layer in range(num_hidden_layers):
            x = Dense(num_hidden_units)(x)
            x = Activation("relu")(x)
        x = Dense(num_labels, activation = "softmax", name = "predictions")(x)

        self.model = self.model = keras.models.Model(input_layer, x, name=self.model_name)
        