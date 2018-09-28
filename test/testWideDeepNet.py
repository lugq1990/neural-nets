# -*- coding:utf-8 -*-
import numpy as np
from sklearn.datasets import load_iris
from neuralNets.wideDeepNets import wideDeepNet
from test.eval import e

iris = load_iris()
x, y = iris.data, iris.target
x = x.reshape(-1, 2, 2)
x_m, y_m = x, y
x_b, y_b = x[:100], y[:100]
x_r, y_r = np.random.random((1000, 2, 2)), np.random.random((1000,))

### multi-class
# model_m = wideDeepNet(3, 2, 2, layer_concat=False)
# e(model_m, x_m, y_m)

### binary-class
# model_b = wideDeepNet(2, 2, 2)
# e(model_b, x_b, y_b)

### regression
model_r = wideDeepNet(-1, 2, 2)
e(model_r, x_r, y_r)