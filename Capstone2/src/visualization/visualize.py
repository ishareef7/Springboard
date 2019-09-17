from sklearn.metrics import confusion_matrix
from mlxtend.plotting import plot_confusion_matrix

import matplotlib.pyplot as plt
import numpy as np


def get_confusion_matrix(y_true, y_pred, normalize=False, title=None, cmap=None):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    labels_ = ['', 'International', 'Pop', 'Rock', 'Electronic', 'Folk', 'Hip-Hop', 'Experimental', 'Instrumental']

    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    if normalize:
        fig, ax = plot_confusion_matrix(conf_mat=cm, colorbar=True, show_absolute=False, show_normed=True, figsize = (7,7),
                                        cmap=cmap)
        ax.set_yticklabels(labels_)
        ax.set_xticklabels(labels_, rotation=45)
        ax.set_title(title, fontsize=16)

    else:
        fig, ax = plot_confusion_matrix(conf_mat=cm, colorbar=True, show_absolute=True, show_normed=False, figsize = (7,7),
                                        cmap=cmap)
        ax.set_yticklabels(labels_)
        ax.set_xticklabels(labels_, rotation=45)
        ax.set_title(title, fontsize = 16)



    return fig, ax
