import datasets
import regression
import numpy as np

X, Y = datasets.load_nonlinear_example1()
ex_X = datasets.polynomial3_features(X)
model = regression.RidgeRegression(alpha=0.0)
model2 = regression.RidgeRegression(alpha=0.1)
model3 = regression.RidgeRegression(alpha=0.5)
model4 = regression.RidgeRegression(alpha=1.0)
model5 = regression.RidgeRegression(alpha=10.0)

model.fit(ex_X, Y)
model2.fit(ex_X, Y)
model3.fit(ex_X, Y)
model4.fit(ex_X, Y)
model5.fit(ex_X, Y)

samples = np.arange(0, 4, 0.1)
x_samples = np.c_[ np.ones(len(samples)), samples ]
ex_x_samples = datasets.polynomial3_features(x_samples)

import matplotlib.pyplot as plt
plt.scatter(X[:,1], Y)
plt.plot(samples, model.predict(ex_x_samples), label = "alpha = 0")
plt.plot(samples, model2.predict(ex_x_samples), label = "alpha = 0.1")
plt.plot(samples, model3.predict(ex_x_samples), label = "alpha = 0.5")
plt.plot(samples, model4.predict(ex_x_samples), label = "alpha = 1.0")
plt.plot(samples, model5.predict(ex_x_samples), label = "alpha = 10.0")
plt.legend()
plt.show()