import tensorflow.keras as keras

MOMENTUM_PARAMETER = 0.9
INITIAL_LEARNING_RATE = 0.01
DECAY_FACTOR_PER_EPOCH = 0.95


class MiniInceptionV3:
    def __init__(self, input_shape, num_labels=10):
        input_layer = keras.layers.Input(shape = input_shape)

        x = keras.layers.Conv2D(filters = 96, kernel_size = (11, 11), strides = (4,4), padding = 'valid')
                

# Eight layers with weights

# Five convolutional, three fully connected

# Output of the last one is 10-way softmax

# multinomial logistic reression objective

