from tensorflow.image import per_image_standardization
from tensorflow.keras.utils import to_categorical

def normalize(x_input):
    return x_input/255

def crop(x_input):
    return x_input[:, 2:-2, 2:-2, :]

def preprocess_input(x_input):
    '''
    Do not casually run this on a laptop. It will crash your whole computer and it will be your fault.
    '''
    x_out = normalize(x_input)
    x_out = crop(x_out)
    x_out = per_image_standardization(x_out)
    return x_out

def preprocess_labels(y_input):
    return to_categorical(y_input)