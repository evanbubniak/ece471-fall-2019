import matplotlib.pyplot as plt
import pandas as pd
import os
from math import ceil

def plot_results(batch_size, num_samples):
    label_markers = ['true', 'random_labels', 'shuffled', 'random_pixels', 'gaussian']

    datasets = []
    for label in label_markers:
        label_data = pd.DataFrame()
        for file in os.listdir(os.getcwd()):
            if ".csv" in file and label in file:
                path = os.path.join(os.getcwd(), file)
                data = pd.read_csv(path)
                label_data = label_data.append(data, sort = False)
        datasets.append(data)

    steps_per_epoch = ceil(num_samples / batch_size)

    for dataset in datasets:
        dataset["thousand steps"] = ((dataset["epoch"] + 1)*steps_per_epoch)/1000

    true_label_format = [{'c': "blue", 'marker': 's', 'edgecolors': 'black'},
                    {'c': "blue"}]

    random_label_format = [{'c': "red", 'marker': 'o', 'edgecolors': 'black'},
                        {'c': "red"}]

    shuffled_pixel_format = [{'c': '#00ff00', 'marker': '*', 'edgecolors': 'black'},
                        {'c': '#00ff00'}]

    random_pixel_format = [{'c': None, 'marker': None, 'edgecolors': None},
                        {'c': "#D742F4"}]

    gaussian_format = [{'c': 'black', 'marker': 'D', 'edgecolors': 'black'},
                        {'c': "black"}]

    formats = [true_label_format, random_label_format, shuffled_pixel_format, random_pixel_format, gaussian_format]

    plt.figure(figsize=(5,5))
    i = 0
    for dataset, data_format in zip(datasets, formats):
        z = 5 if i == 3 else 0
        if i != 3:
            plt.scatter(dataset["thousand steps"].values[1:],
                        dataset["loss"].values[1:],
                        zorder = 10,
                        **data_format[0])
        plt.plot(dataset["thousand steps"].values[1:],
                dataset["loss"].values[1:],
                zorder = z,
                **data_format[1]
            )

        i+=1

    plt.legend(['true labels', 'random labels', 'shuffled pixels', 'random pixels', 'gaussian'])
    plt.xlabel("thousand steps")
    plt.ylabel("average_loss")
    plt.xlim(0, 25)
    plt.ylim(0, 2.5)
    plt.savefig("output.eps")
    plt.savefig("output.png")


# Need input data [training_step_num, average_loss] for each one.

# Plot of learning curves
# all dots have black outline
#true labels: blue, with blue square dots

# random labels - red, with red circular dots

# shuffled pixels: green, with green star 5-point star dots

# random pixels: purple, with no dots

# gaussian: black, with black diamond dots

# x-label: thousand steps (steps of 5, from 0 to 25)
# y-label: average_loss (steps of 0.5, from 0 to 2.5)
# dots evenly spaced along x axis, with about 1.3 dots per training step, starting at about 0.5
