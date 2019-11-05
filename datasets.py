import numpy as np

def load_linear_example1():
    """
    >>> import datasets
    >>> x,y = datasets.load_linear_example1()
    >>> print(x[0])
    [1 4]
    >>> print(y)
    [ 7 10 11 14]
    """
    X = np.array([[1,4],[1,8],[1,13],[1,17]])
    Y = np.array([7,10,11,14])
    return X,Y

x,y = load_linear_example1()
print(y)