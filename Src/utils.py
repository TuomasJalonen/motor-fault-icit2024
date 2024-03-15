"""
Author: Tuomas Jalonen

This file contains utility functions.
"""

# pylint: disable=import-error
import os
import time
from pathlib import Path
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras import layers
from tensorflow.keras.layers import Dense, Flatten, Conv1D, MaxPooling1D
from tensorflow.keras.layers import BatchNormalization, ReLU, Dropout
from tensorflow.keras.regularizers import l2

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.font_manager
from matplotlib import rc
import numpy as np
import seaborn as sn


def lr_schedule(epoch):
    """Learning Rate Schedule

    Learning rate is scheduled to be reduced after certain epochs.
    Called automatically every epoch as part of callbacks during training.

    # Arguments
        epoch (int): The number of epochs

    # Returns
        lr (float32): learning rate
    """
    learning_rate = 1e-5
    if epoch > 100:
        learning_rate *= 1e-1
    # elif epoch > 40:
    #     learning_rate *= 1e-2
    # elif epoch > 100:
    #     learning_rate *= 1e-1

    return learning_rate


def save_model(directory):
    """
    This function is a Keras callback to save the model.
    """
    acc_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=os.path.join(directory, "saved_model"),
        save_weights_only=True,
        monitor="val_accuracy",
        mode="max",
        save_best_only=False,
    )

    return acc_callback


class TimingCallback(tf.keras.callbacks.Callback):
    def __init__(self, logs={}):
        self.logs = []

    def on_epoch_begin(self, epoch, logs={}):
        self.starttime = time.time()

    def on_epoch_end(self, epoch, logs={}):
        self.logs.append(time.time() - self.starttime)


def cnn_model(input_shape: list, n_blocks: int, n_classes: int):
    model = Sequential()
    # Preliminary
    model.add(
        Conv1D(64, (5,), activation="relu", padding="same", input_shape=input_shape)
    )
    model.add(MaxPooling1D(pool_size=(2), strides=2))

    if n_blocks >= 1:
        for i in range(n_blocks):
            model.add(Conv1D(64, (3,), activation="relu", padding="same"))
            model.add(MaxPooling1D(pool_size=(2), strides=2))

    # Fully Connected
    model.add(Flatten())
    model.add(Dense(128, activation="relu"))
    model.add(Dropout(0.4))
    model.add(Dense(n_classes, activation="softmax"))

    return model


def matplotlib_latex_font():
    """
    This function sets a latex font to be used in matplotlib figures.
    """
    # Restore Matplotlib settings
    matplotlib.rc_file_defaults()

    # Set latex-style font
    fpath = Path(
        matplotlib.get_data_path(),
        # Download the font and edit its path here
        ".../cmunrm.ttf",
    )

    matplotlib.font_manager.fontManager.addfont(fpath)
    prop = matplotlib.font_manager.FontProperties(fname=fpath)

    #  Set it as default matplotlib font
    matplotlib.rc("font", family="sans-serif")
    matplotlib.rcParams.update({"font.sans-serif": prop.get_name()})


def plot_training_curves(directory, history):
    """
    This function plots training curves.
    """

    # Restore Matplotlib settings
    matplotlib.rc_file_defaults()

    # Accuracy curve
    plt.plot(history.history["accuracy"])
    plt.plot(history.history["val_accuracy"])
    plt.ylabel("Accuracy")
    plt.xlabel("Epoch")
    plt.yticks([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    plt.legend(["Training", "Testing"], loc="upper left")
    plt.savefig(os.path.join(directory, "acc.pdf"), dpi=300)
    plt.clf()

    # Loss curve
    plt.plot(history.history["loss"])
    plt.plot(history.history["val_loss"])
    plt.ylabel("Categorical Crossentropy")
    plt.xlabel("Epoch")
    plt.legend(["Training", "Testing"], loc="upper left")
    plt.savefig(os.path.join(directory, "loss.pdf"), dpi=300)
    plt.clf()

    # F1-score curve
    plt.plot(history.history["f1_score"])
    plt.plot(history.history["val_f1_score"])
    plt.ylabel("F1-score")
    plt.xlabel("Epoch")
    plt.yticks([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    plt.legend(["Training", "Testing"], loc="upper left")
    plt.savefig(os.path.join(directory, "f1.pdf"), dpi=300)
    plt.clf()

    # Precision curve
    plt.plot(history.history["precision"])
    plt.plot(history.history["val_precision"])
    plt.ylabel("Precision")
    plt.xlabel("Epoch")
    plt.yticks([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    plt.legend(["Training", "Testing"], loc="upper left")
    plt.savefig(os.path.join(directory, "precision.pdf"), dpi=300)
    plt.clf()

    # Recall curve
    plt.plot(history.history["recall"])
    plt.plot(history.history["val_recall"])
    plt.ylabel("Recall")
    plt.xlabel("Epoch")
    plt.yticks([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    plt.legend(["Training", "Testing"], loc="upper left")
    plt.savefig(os.path.join(directory, "recall.pdf"), dpi=300)
    plt.clf()

    # Combined Accuracy and Loss curve
    fig, ax1 = plt.subplots()

    ax1.plot(history.history["accuracy"])
    ax1.plot(history.history["val_accuracy"])
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Accuracy")
    ax1.set_yticks([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    ax1.set_ylim(0.0, 1.0)
    # ax1.grid()

    ax2 = ax1.twinx()
    ax2.plot(history.history["loss"], "--")
    ax2.plot(history.history["val_loss"], "--")
    ax2.set_ylabel("Categorical Crossentropy")
    ax2.set_yticks([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    ax2.set_ylim(0.0, 1.0)

    fig.legend(
        ["Training Accuracy", "Testing Accuracy", "Training Loss", "Testing Loss"],
        bbox_to_anchor=(0.9, 0.5),
    )
    plt.savefig(os.path.join(directory, "acc_loss.pdf"), dpi=300)
    plt.clf()

    return None


def plot_cm(directory, df_cm):
    """
    This function plots the confusion matrix.
    """
    # matplotlib_latex_font()
    matplotlib.rcParams.update({"font.size": 18})
    ax = sn.heatmap(
        df_cm,
        annot=True,
        annot_kws={"size": 18},
        cmap="YlGnBu",
        fmt=".3f",
        cbar_kws={"ticks": [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]},
        vmin=0,
        vmax=1,
    )
    ax.tick_params(left=False, bottom=False)
    plt.yticks(rotation=0)
    plt.xlabel("Predicted labels")
    plt.ylabel("True labels")
    plt.tight_layout()
    plt.savefig(os.path.join(directory, "cm.pdf"), dpi=300)
    plt.clf()

    return None


def plot_auc(directory, fpr, tpr, auc):
    """
    This function plots the receiving operating characteristics curve with area under the curve.
    """
    # Restore Matplotlib settings
    matplotlib.rc_file_defaults()

    plt.plot(fpr, tpr, label="AUC = {:.3f}".format(auc))
    plt.xlabel("False positive rate")
    plt.ylabel("True positive rate")
    plt.title("ROC curve")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(os.path.join(directory, "roc.pdf"), dpi=300)
    plt.clf()

    return None


def plot_tsne(directory, tsne, true_classes, color_dict):
    """
    This function plots the t-sne.
    """
    matplotlib_latex_font()
    matplotlib.rcParams.update({"font.size": 18})

    # scale and move the coordinates so they fit [0; 1] range
    def scale_to_01_range(x):
        # compute the distribution range
        value_range = np.max(x) - np.min(x)

        # move the distribution so that it starts from zero
        # by extracting the minimal value from all its values
        starts_from_zero = x - np.min(x)

        # make the distribution fit [0; 1] by dividing by its range
        return starts_from_zero / value_range

    # extract x and y coordinates representing the positions of the images on T-SNE plot
    tx = tsne[:, 0]
    ty = tsne[:, 1]

    tx = scale_to_01_range(tx)
    ty = scale_to_01_range(ty)

    sn.scatterplot(
        x=tx,
        y=ty,
        hue=true_classes,
        hue_order=color_dict.keys(),
        palette=color_dict,
        alpha=0.7,
    )

    plt.legend(
        bbox_to_anchor=(0.5, 1.17),
        loc="upper center",
        ncol=4,
        columnspacing=0.17,
        handletextpad=0.17,
    )

    plt.xticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    plt.yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    plt.savefig(os.path.join(directory, "t-sne.pdf"), dpi=300)
    plt.clf()

    return None
