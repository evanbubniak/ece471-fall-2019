'''
The CIFAR10 dataset contains 50,000 training and 10,000 validation images, split into 10 classes.
Each image is of size 32x32, with 3 color channels.
We divide the pixel values by 255 to scale them into [0, 1], crop from the center to get 28x28 inputs,
and then normalize them by subtracting the mean and dividing the adjusted standard deviation independently for each image with the per_image_whitening function in TENSORFLOW

'''

from tensorflow.image import per_image_standardization

def preprocess_data(x_input):
    '''
    Do not casually run this on a laptop. It will crash your whole computer and it will be your fault.
    '''
    x_out = x_input/255
    x_out = x_out[:, 2:-2, 2:-2, :]
    x_out = per_image_standardization(x_out)
    return x_out
