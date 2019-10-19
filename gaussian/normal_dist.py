import numpy as np
import matplotlib.pyplot as plt
np.random.seed(31415)

# constants


xs = np.random.uniform(low = 0, high = 1, size = n)
epsilons = np.random.normal(loc = 0, scale = sigma_noise, size = n)
ys = [np.sin(2*np.pi*x) + epsilon for x, epsilon in zip(xs, epsilons)]
plt.scatter(xs, ys)
plt.savefig("plot.png")
