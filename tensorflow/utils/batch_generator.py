import numpy as np


def batch_generator(X, y, batch_size):
    """Primitive batch generator
        """
    size = X.shape[0]
    # X_copy = X.copy()
    # y_copy = y.copy()
    # indices = np.arange(size)
    # X_copy = X_copy[indices]
    # y_copy = y_copy[indices]
    i = 0
    while True:
        if i + batch_size <= size:
            yield X[i:i + batch_size], y[i:i + batch_size]
            i += batch_size
        else:
            i = 0


if __name__ == "__main__":
    # Test batch generator
    gen = batch_generator(np.array([['a', 'e'], ['b', 'f'],
                                   ['c', 'g'], ['d', 'h']]), np.array([1, 2, 3, 4]), 2)
    for _ in range(8):
        xx, yy = next(gen)
        print(xx, yy)
