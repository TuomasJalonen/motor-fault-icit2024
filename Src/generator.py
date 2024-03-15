"""
Author: Tuomas Jalonen

This file contains a custom Keras generator, which reads .mat-files.
"""

import os
import tensorflow
import numpy as np
from scipy.io import loadmat


class DataGenerator(tensorflow.keras.utils.Sequence):
    "Generates data for Keras"

    def __init__(
        self,
        list_IDs,
        labels,
        datadir,
        datachannel,
        batch_size,
        dim,
        n_channels,
        n_classes,
        shuffle=True,
        sensor_index=None,
    ):
        "Initialization"
        self.dim = dim
        self.batch_size = batch_size
        self.labels = labels
        self.datadir = datadir
        self.datachannel = datachannel
        self.list_IDs = list_IDs
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.sensor_index = sensor_index
        self.on_epoch_end()

    def __len__(self):
        "Denotes the number of batches per epoch"
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        "Generate one batch of data"
        # Generate indexes of the batch
        indexes = self.indexes[index * self.batch_size : (index + 1) * self.batch_size]
        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]
        # list_IDs_temp = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        # Generate data
        X, y = self.__data_generation(list_IDs_temp)

        return X, y

    def on_epoch_end(self):
        "Updates indexes after each epoch"
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp):
        "Generates data containing batch_size samples"  # X : (n_samples, *dim, n_channels)
        # Initialization
        X = np.empty((self.batch_size, *self.dim, self.n_channels))
        y = np.empty((self.batch_size), dtype=int)

        # Generate data
        for i, ID in enumerate(list_IDs_temp):

            if self.n_channels == 1 and self.datachannel == "x":
                X[i,] = np.expand_dims(
                    np.array(
                        loadmat(
                            os.path.join(
                                self.datadir, ID.split("_", 1)[0], ID.split("_", 1)[1]
                            )
                        )[self.datachannel]
                    )[:, self.sensor_index],
                    axis=-1,
                )
            else:
                X[i,] = np.array(
                    loadmat(
                        os.path.join(
                            self.datadir, ID.split("_", 1)[0], ID.split("_", 1)[1]
                        )
                    )[self.datachannel]
                )

            X[i,] = (X[i,] - np.mean(X[i,])) / (np.std(X[i,]))

            # Store class
            y[i] = self.labels[ID]

        return X, tensorflow.keras.utils.to_categorical(y, num_classes=self.n_classes)
