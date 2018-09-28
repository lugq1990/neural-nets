# -*- coding:utf-8 -*-
import os
from sklearn.model_selection import train_test_split
from utils.utils import plot_acc_loss, check_label

# Model fitting function, here this function will split data to be train and validation data sets.
# Model training will use train data, and will also evaluate after training model accuracy.
def fit(model, data, label, n_classes, epochs=100, batch_size=128, callback=None, silence=False):
    label = check_label(label, n_classes)
    xtrain, xvalidate, ytrain, yvalidate = train_test_split(data, label, test_size=.2, random_state=1234)

    # If no callback is called, then just use EarlyStopping for validation data
    # Here changed if callback is None, then just not use callback.
    # if callback is None:
    #     callback = EarlyStopping(monitor='val_loss', patience=patience)
    if callback is None:
        his = model.fit(xtrain, ytrain, epochs=epochs, batch_size=batch_size,
                                  validation_data=(xvalidate, yvalidate), verbose=1)
    else:
        his = model.fit(xtrain, ytrain, epochs=epochs, batch_size=batch_size,
                                  validation_data=(xvalidate, yvalidate),callback=[callback], verbose=1)
    if not silence:
        val_acc = model.evaluate(xvalidate, yvalidate, batch_size=1024)[1]
        print('After training, model evaluate on validate data accuracy = {:.2f}%'.format(val_acc * 100))

    return model, his


# This if model evaluation function to evaluate test data sets after model training.
def evaluate(model, data, label, n_classes, batch_size=1024):
    label = check_label(label, n_classes)
    eval_acc = model.evaluate(data, label, batch_size=batch_size)[1]

    print('Model evaluate on Test Datasets accuracy = {:.2f}%'.format(eval_acc * 100))
    return eval_acc


# This is prediction function for after model training to predicte un-seen data sets.
# Can use which step to be used for prediction.
def predict(model, data, batch_size=1024, step=None):
    return model.predict(data, batch_size=batch_size, steps=step)


# save model to disk.
def save_model(model, path=None, model_name=None):
    if path is None:  # If path is not given, save model to current directory.
        path = os.getcwd()
    if model_name is None:
        model_name = 'trained_model.hd5'
    model.save(path + model_name)
    print('Model have been saved to disk path: ', path + model_name)


# Plot accuracy and loss curve for evaluating model training steps.
def plot_acc(his, n_classes, plot_acc=True, plot_loss=True, figsize=(8, 6)):
    # For regression problem, there is no accuracy.
    if n_classes == -1:
        plot_acc = False
    plot_acc_loss(his, plot_acc, plot_loss, figsize=figsize)
