# -*- coding:utf-8 -*-
import numpy as np
import tensorflow.keras as keras
import matplotlib.pyplot as plt
from matplotlib import style

# This is used for converting 1D label to nD one-hot label
def check_label(label, num_classes):
    if len(label.shape) == 1 and num_classes >=2:
        return keras.utils.to_categorical(label, num_classes=len(np.unique(label)))
    return label

# This function is used for checking model construction parameters, different dimensions.
def check_dims(dims, is_2d, is_3d):
    if is_2d and dims is None:
        return '2D'
    elif is_3d or dims is not None:
        return '3D'

# This is used for ploting accuracy and loss curve
def plot_acc_loss(his, plot_acc=True, plot_loss=True, figsize=(8, 6)):
    style.use('ggplot')

    if plot_acc:
        fig_1, ax_1 = plt.subplots(1, 1, figsize=figsize)
        ax_1.plot(his.history['acc'], label='Train Accuracy')
        ax_1.plot(his.history['val_acc'], label='Validation Accuracy')
        ax_1.set_title('Train & Validation Accuracy curve')
        ax_1.set_xlabel('Epochs')
        ax_1.set_ylabel('Accuracy score')
        plt.legend()

    if plot_loss:
        fig_2, ax_2 = plt.subplots(1, 1, figsize=figsize)
        ax_2.plot(his.history['loss'], label='Train Loss')
        ax_2.plot(his.history['val_loss'], label='Validation Loss')
        ax_2.set_title('Train & Validation Loss curve')
        ax_2.set_xlabel('Epochs')
        ax_2.set_ylabel('Loss score')
        plt.legend()