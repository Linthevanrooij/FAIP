import numpy as np
from sklearn.datasets import make_moons


def create_dataset(size, dimension, delta):
    #Size of the dataset
    n = (int(size / 2), int(size / 2))

    # Dimension of the dataset
    dim = dimension

    # Distance between the mean of the classes
    delta = delta

    # Prior probabilities per class
    p = [0.5, 0.5]  # make a list
    n = np.asarray(n)
    p = np.asarray(p)
    c = len(p)
    P = np.cumsum(p)
    P = np.concatenate((np.zeros(1), P))
    N = np.zeros(c, dtype=int)

    x = np.random.rand(np.sum(n))
    for i in range(0, c):
        N[i] = np.sum((x > P[i]) & (x < P[i + 1]))
    x0 = np.random.randn(N[0], dim)
    x1 = np.random.randn(N[1], dim)
    x1[:, 0] = x1[:, 0] + delta  # move data from class 1
    x_features = np.concatenate((x0, x1), axis=0)

    # labels
    lab = (0, 1)
    y_out = np.repeat(lab[0], N[0])
    for i in range(1, len(N)):
        y_out = np.concatenate((y_out, np.repeat(lab[i], N[i])))
    return x_features, y_out


def get_moon_data(size):
    X, y = make_moons(size, noise=0.2)
    return X, y

