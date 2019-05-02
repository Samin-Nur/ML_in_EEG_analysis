from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import numpy as np
import itertools


def plot_confusion_matrix(pred_class, actual_class,
                          title='Confusion matrix',
                          size=None):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.

    Code from: http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html
    """
    cm = confusion_matrix(actual_class, pred_class)



    cmap = plt.cm.Blues
    cm = cm.astype('float') / np.sum(cm, axis=1, keepdims=True)
    cm = np.nan_to_num(cm)

    if cm is not None:
        cm = cm[0:size, 0:size]

    plt.figure(title)
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    x = np.arange(1, 6, 1)  # the grid to which your data corresponds
    nx = x.shape[0]
    no_labels = 5  # how many labels to see on axis x
    step_x = int(nx / (no_labels - 1))  # step between consecutive labels
    x_positions = np.arange(0, nx, step_x)  # pixel count at label position
    x_labels = x[::step_x]  # labels you want to see
    plt.xticks(x_positions, x_labels)
    plt.yticks(x_positions,x_labels)

    fmt = '.2f'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True class')
    plt.xlabel('Predicted class')
