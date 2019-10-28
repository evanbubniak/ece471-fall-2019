import argparse

model_names = ["MiniInceptionV3", "MiniInceptionV3_without_BatchNorm", "AlexNet", "MLP_1x512", "MLP_3x512"]

parser = argparse.ArgumentParser()
parser.add_argument("--model", nargs="*", default = model_names)
args = parser.parse_args()
print(args)
