import numpy as np
import matplotlib.pyplot as plt

def randomize_labels(num_labels, num_unique_labels):
    random_labels = np.random.randint(low=0, high=num_unique_labels, size=num_labels)
    categorical_random_labels = np.eye(num_unique_labels)[random_labels]
    return categorical_random_labels

def shuffle_pixels(normalized_pixel_data):
    shuffled_pixel_data = normalized_pixel_data
    np.random.shuffle(shuffled_pixel_data)
    return shuffled_pixel_data

def randomize_pixels(normalized_pixel_data):
    randomized_pixel_data = np.random.uniform(low = 0, high = 1.0, size=(normalized_pixel_data.shape))
    return randomized_pixel_data

def gaussian_noise(normalized_pixel_data):
    noise = np.random.normal(0,1,normalized_pixel_data.shape)
    noisy_pixel_data = normalized_pixel_data + noise
    np.clip(noisy_pixel_data, a_min=0, a_max = 1, out=noisy_pixel_data)
    return noisy_pixel_data

#garbage_pixel_data = np.array([[[0.2,0.3,0.5],[0.3,0.2,0.5],[0.6,1,0.4]],[[0.4,0.9,0.7],[0.1,0.2,0.3],[0.4,0.5,0.6]]])
garbage_pixel_data = np.random.uniform(low=0,high=1,size=(10, 32, 32, 3))
randomized_pixel_data = randomize_pixels(garbage_pixel_data)
shuffled_pixel_data = shuffle_pixels(garbage_pixel_data)
gaussian_pixel_data = gaussian_noise(garbage_pixel_data)

print(gaussian_pixel_data)