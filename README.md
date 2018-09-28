# neuralNets

Build advanced deep learning models in one line. Most quickest and easiest way to build deep learning models.

## Getting Started

In recent years, most popular technology in AI is Deep Learning. It has showed it power in many domains, such as image, natual language processing, voice and so on. If you want to use AI models for your goals to be solved better, use deep learning! If you find many machine learning predicted not well, use deep learning!

So deep learning can do lots of works, but how to use it? Thanks to Google teams, there is a great way to build models by using TensorFlow! This is based on TensorFlow to build many deep learning models, such as basic: # DNN, # RNN, # CNN, also with some advanced and more powerful model structure, such as: # ResidualNet, # DenseNet, # LSTM, # GRU and # Wide&Deep. And in machine learning domain, there are two main categories to be solved: Classification(binary, multiclass) and regression. They are all supported, you only need to change one parameter to rebuild your models. Great.

### Installing

Git remote repository or clone source code to disk, in neuralNets directory:

```
python setup.py install
```

### OKay, show some examples how to use.

```python
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from neuralNets.dnnNets import dnnNet

iris = load_iris()
x, y = iris.data, iris.target
xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=.2)

model = dnnNet(3, 4, n_layers=2, n_units=64, use_batch=False)
model.fit(xtrain, ytrain, epochs=200)
test_acc = model.evaluate(xtest, ytest)
pred = model.predict(xtest)
model.plot_acc()
```
![acc_curve](image/acc.png?raw=true)
![loss_curve](image/loss.png?raw=true)

Easy?

Any problems are welcome!

## Contributing

All contributions or issues are welcome!

## Authors

* **lugq** - *Initial work* - [lugq1990](https://github.com/lugq1990)

### Paper links:
LSTM:https://www.isca-speech.org/archive/archive_papers/interspeech_2012/i12_0194.pdf
GRU:https://arxiv.org/pdf/1412.3555.pdf
GoogleNet:https://www.cs.unc.edu/~wliu/papers/GoogLeNet.pdf
ResidualNet:https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/He_Deep_Residual_Learning_CVPR_2016_paper.pdf

### Happy Deep Learning Modeling.
