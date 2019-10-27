import os, os.path
from matplotlib.image import imread

PNGPath = os.path.join(os.getcwd(),"dataset", "images", "mnist_png", "testing", "0", "3.png")

def get_nparray_from_png(PNGPath):
    img = imread(PNGPath)
    return img