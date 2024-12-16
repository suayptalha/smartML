import numpy as np

def train_test_split(x : np.array, y, random_state : int, test_size : int) -> tuple:
    np.random.seed(random_state)
    
    indices = np.arange(x.shape[0])

    np.random.shuffle(indices)

    test_size_int = int(test_size * x.shape[0])

    train_indices = indices[:test_size_int]
    test_indices = indices[test_size_int:]

    x_train = x[train_indices]
    y_train = y[train_indices]
    x_test = x[test_indices]
    y_test = y[test_indices]

    return (x_train, y_train, x_test, y_test)
