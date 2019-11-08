import numpy as np

class LinearRegression:
    """
    >>> import regression
    >>> model = regression.LinearRegression()
    >>> model.x
    >>> #nothing
    """
    x = None
    theta = None
    y = None

    def fit(self,x,y):
        """
        >>> import regression
        >>> import datasets
        >>> import importlib
        >>> x,y = datasets.load_linear_example1()
        >>> importlib.reload(regression)
        <module 'regression' from '/Users/e175768/temp/regression-test/regression.py'>
        >>> model = regression.LinearRegression()
        >>> model.fit(x,y)
        >>> model.theta
        array([5.30412371, 0.49484536])
        """
        temp = np.linalg.inv(np.dot(x.T,x))
        self.theta = np.dot(np.dot(temp,x.T),y)


    def predict(self,x):
        """
        >>> import datasets
        >>> import regression
        >>> import importlib
        >>> x,y = datasets.load_linear_example1()
        >>> importlib.reload(regression)
        <module 'regression' from '/Users/e175768/temp/regression-test/regression.py'>
        >>> model = regression.LinearRegression()
        >>> model.fit(x,y)
        >>> model.predict(x)
        array([ 7.28350515,  9.2628866 , 11.7371134 , 13.71649485])
        """
        return np.dot(x,self.theta)

    def score(self,x,y):
        """
        >>> import datasets
        >>> import regression
        >>> import importlib
        >>> x,y = datasets.load_linear_example1()
        >>> importlib.reload(regression)
        <module 'regression' from '/Users/e175768/temp/regression-test/regression.py'>
        >>> model = regression.LinearRegression()
        >>> model.fit(x,y)
        >>> model.score(x,y)
        1.2474226804123711
        """
        error = self.predict(x) - y
        return (error**2).sum()
