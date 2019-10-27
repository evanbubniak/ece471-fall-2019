import os, os.path
import tarfile

mnist_path = os.path.join(os.getcwd(), "dataset")

def fetch_mnist_data(mnist_path=mnist_path):
    mnist_tar = tarfile.open(os.path.join(mnist_path, "MNIST_png.tar"))
    mnist_tar.extractall(path=os.path.join(mnist_path, "images"))
    mnist_tar.close()

fetch_mnist_data()