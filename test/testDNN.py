# -*- coding:utf-8 -*-
import numpy as np
from sklearn.datasets import load_iris
from neuralNets.dnnNets import dnnNet
from test.eval import e

iris = load_iris()
x_m, y_m = iris.data, iris.target
x_b, y_b = iris.data[:100], iris.target[:100]
x_r = np.random.random((1000, 10))
y_r = np.random.random((1000))

### multi-classes
model_m = dnnNet(3, 4, n_units=128, use_batch=False)
e(model_m, x_m, y_m)

### binary-class
# model_b = dnnNet(2, 4)
# e(model_b, x_b, y_b)

### regression
# model_r = dnnNet(-1, 10 )
# e(model_r, x_r, y_r)
