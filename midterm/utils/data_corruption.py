import numpy as np
np.random.seed(31415)

def randomize_labels(num_labels, num_unique_labels):
    random_labels = np.random.randint(low=0, high=num_unique_labels, size=num_labels)
    categorical_random_labels = np.eye(num_unique_labels)[random_labels]
    return categorical_random_labels

def shuffle_image_pixels(image_data):
    shuffled_image_pixels = image_data
    np.random.shuffle(shuffled_image_pixels)
    np.random.seed(31415)
    return shuffled_image_pixels

def shuffle_pixels(normalized_pixel_data):
    shuffled_pixel_data = normalized_pixel_data
    np.apply_along_axis(shuffle_image_pixels, 0, shuffled_pixel_data)
    return shuffled_pixel_data

def randomize_pixels(normalized_pixel_data):
    randomized_pixel_data = np.random.uniform(low = 0, high = 1.0, size=(normalized_pixel_data.shape))
    return randomized_pixel_data

def create_gaussian_noise(normalized_pixel_data):
    pixel_data = normalized_pixel_data.numpy().astype('float32')
    mean = pixel_data.mean(axis=0)
    std = pixel_data.std(axis=0)
    noisy_pixel_data = np.random.normal(mean,std,normalized_pixel_data.shape)
    np.clip(noisy_pixel_data, a_min=0, a_max = 1, out=noisy_pixel_data)
    return noisy_pixel_data