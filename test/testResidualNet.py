# -*- coding:utf-8 -*-
import numpy as np
from sklearn.datasets import load_iris
from test.eval import e

iris = load_iris()
x, y = iris.data, iris.target
x = x.reshape(-1, 2, 2)
x_m, y_m = x, y
x_b, y_b = x[:100], y[:100]
x_r = np.random.random((1000, 2, 2))
y_r = np.random.random((1000))

### multi-class
# model_m = residualNet(3, 2, 2)
# e(model_m, x_m, y_m)

### binary-class
# model_b = residualNet(2, 2, 2)
# e(model_b, x_b, y_b)

### regression
# model_r = residualNet(-1, 2, 2)
# e(model_r, x_r, y_r)

from neuralNets.residualNets import residualNet
model_r = residualNet(-1, 2, 2)
e(model_r, x_r, y_r)