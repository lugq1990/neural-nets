# -*- coding:utf-8 -*-
import numpy as np
from sklearn.datasets import load_iris
from neuralNets.rnnNets import rnnNet
from test.eval import e


iris = load_iris()
x, y = iris.data, iris.target
x = x.reshape(-1, 2, 2)
x_m, y_m = x, y
x_b, y_b = x[:100], y[:100]
x_r = np.random.random((1000, 2, 2))
y_r = np.random.random((1000))

### multi-class
# model_m = rnnNet(3, 2, 2)
# e(model_m, x_m, y_m)

### binary
# model_b = rnnNet(2, 2, 2)
# e(model_b, x_b, y_b)

### regression
model_r = rnnNet(-1, 2, 2)
e(model_r, x_r, y_r)