"""
Plot of learning curves
true labels: blue, with blue square dots
random labels - red, with red circular dots
shuffled pixels: green, with green star 5-point star dots
random pixels: purple, with no dots
gaussian: black, with black diamond dots
"""
import matplotlib.pyplot as plt
import pandas as pd
import os
import argparse
from math import ceil

label_markers = ['true', 'random_labels', 'shuffled', 
                 'random_pixels', 'gaussian']
model_names = ["MiniInceptionV3", 
               "MiniInceptionV3_without_BatchNorm",
               "AlexNet", "MLP_1x512", "MLP_3x512"]
model_correspondence = {
    "MiniInceptionV3": 1,
    "MiniInceptionV3_without_BatchNorm":2,
    "AlexNet":3,
    "MLP_1x512":4,
    "MLP_3x512":5
}

def get_max_folder_num():
    everything_in_dir = os.listdir(os.getcwd())
    folders_in_dir = filter(lambda f: os.path.isdir(f) and 
                            "output" in f, everything_in_dir)
    max_folder_num = 1
    for folder in folders_in_dir:
        folder_num = folder[(folder.find("_") + 1):]
        max_folder_num = max(int(folder_num), max_folder_num)
    return max_folder_num


parser = argparse.ArgumentParser()
parser.add_argument("--model", nargs="*", default = model_names)
parser.add_argument("-e", "--num_epochs",
    nargs="?", type=int, default=100)
parser.add_argument("-b", "--batch_size",
    nargs="?", type=int, default=100)
parser.add_argument("--output_num", nargs = "?",
    default = get_max_folder_num())

parser.add_argument("-i", "--iterate", action = "store_true")
args = parser.parse_args()

OUTPUT_DIR = "output_{}".format(args.output_num)

def plot_results(steps_per_epoch, models = model_names, 
                 plot_name="All_Models_Averaged"):
    def filter_dir_files(file_name):
        conditionals = [".csv" in file_name]
        if ("MiniInceptionV3" in models and
            "MiniInceptionV3_without_BatchNorm" not in models and
            "BatchNorm" in file_name):
            return False
        else:
            conditionals.append(any([model_name in file_name 
                for model_name in models]))
        return all(conditionals)

    all_data = []


    all_files_in_dir = list(filter(filter_dir_files, 
                                   os.listdir(OUTPUT_DIR)))

    for label in label_markers:
        data_by_model = []
        for file_name in all_files_in_dir:
            if label in file_name:
                path = os.path.join(OUTPUT_DIR, file_name)
                data = pd.read_csv(path)
                data = data[:args.num_epochs]
                data_by_model.append(data)
        all_data.append(data_by_model)

    all_losses = []
    for data_corruption in all_data:
        avg_loss = pd.DataFrame({'average loss': 
                                 [0] * len(data_corruption[0])})
        avg_loss['epoch'] = avg_loss.index
        for model_results in data_corruption:
            avg_loss['average loss'] = avg_loss['average loss'] + \
                model_results["loss"]
        avg_loss['average loss'] = avg_loss['average loss'] / \
            len(data_corruption)
        all_losses.append(avg_loss)
    for dataset in all_losses:
        dataset["thousand steps"] = \
            ((dataset["epoch"] + 1)*steps_per_epoch)/1000
    true_label_format = [{'c': "blue", 'marker': 's', 
                          'edgecolors': 'black'},
                        {'c': "blue"}]

    random_label_format = [{'c': "red", 'marker': 'o', 
                            'edgecolors': 'black'},
                        {'c': "red"}]

    shuffled_pixel_format = [{'c': '#00ff00', 'marker': '*', 
                              'edgecolors': 'black'},
                        {'c': '#00ff00'}]

    random_pixel_format = [{'c': None, 'marker': None, 
                            'edgecolors': None},
                        {'c': "#D742F4"}]

    gaussian_format = [{'c': 'black', 'marker': 'D', 
                        'edgecolors': 'black'},
                        {'c': "black"}]

    formats = [true_label_format, random_label_format, 
               shuffled_pixel_format, random_pixel_format, 
               gaussian_format]
    legend_names = ['true labels', 'random labels', 
                    'shuffled pixels', 'random pixels', 'gaussian']
    
    fig1 = plt.figure(figsize=(3,3))
    i = 0
    array_of_linmarks=[]
    for dataset, data_format in zip(all_losses, formats):
        lin = None
        mark = None
        ax = plt.gca()
        z = 5 if i == 3 else 0
        if i != 3:
            mark = ax.scatter(dataset["thousand steps"].values[1:],
                        dataset["average loss"].values[1:],
                        s = 15,
                        zorder = 10,
                        **data_format[0])
        lin, = ax.plot(dataset["thousand steps"].values[1:],
                dataset["average loss"].values[1:],
                zorder = z,
                **data_format[1]
            )
        if mark:
            array_of_linmarks.append((lin, mark))
        else:
            array_of_linmarks.append((lin))
        i+=1
    ax.tick_params(axis = 'both', direction = 'in', 
                   top = True, right = True)
    leg = ax.legend(array_of_linmarks, legend_names, 
                    scatterpoints=2, framealpha = 1, 
                    scatteryoffsets=[0.5],
					loc = 'upper right')
    leg.set_zorder(20)
    leg.get_frame().set_facecolor('w')
    ax.set_xticks([0, 5, 10, 15, 20, 25])
    ax.set_yticks([0, 0.5, 1, 1.5, 2.0, 2.5])

    plt.draw()

    plt.xlabel("thousand steps")
    plt.ylabel("average_loss")
    plt.title(plot_name)
    plt.xlim(0, 25)
    plt.ylim(0, 2.5)
    plt.tight_layout()
    print(os.path.join(OUTPUT_DIR, "{}.eps".format(plot_name)))
    print(os.path.join(OUTPUT_DIR, "{}.png".format(plot_name)))
    fig1.savefig(os.path.join(OUTPUT_DIR, "{}.eps".format(plot_name)))
    fig1.savefig(os.path.join(OUTPUT_DIR, "{}.png".format(plot_name)))

if __name__ == "__main__":
    if args.iterate:
        for model in args.model:
            plot_results(
                ceil(50000/args.batch_size),
                models = [model],
                plot_name = model)
    else:
        plot_results(
            ceil(50000/args.batch_size),
            models = args.model)