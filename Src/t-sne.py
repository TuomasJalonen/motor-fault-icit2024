#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Author: Tuomas Jalonen

This script is used for creating the t-sne results.
"""
# pylint: disable=import-error
import os
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Flatten
import numpy as np
from sklearn.manifold import TSNE
from sklearn.model_selection import KFold


from generator import DataGenerator
from utils import plot_tsne, cnn_model

models = {
    "cnn2": {
        "input_size": (2000, 2),
        "n_channels": 2,
        "sensor_index": None,
        "number_of_blocks": "4",
        "datachannel": "s",
    },
}

# Parameters
params = {
    "dim": (2000,),
    "batch_size": 100,
    "n_classes": 4,
    "shuffle": True,
}
params_val = params
params_val["shuffle"] = False
epochs = 150
layer_index = -3

# Base_dir is your directory which contains the "Dataset" and "Results" folders
base_dir = "..."

noise_levels = [
    "Clean",
    "-5dB",
    "0dB",
    "5dB",
    # "10dB",
    # "15dB",
    # "20dB",
]
conditions = {"Normal": 0, "Outer": 1, "Inner": 2, "Ball": 3}
kf = KFold(n_splits=5)

# loop noise levels
for noise_level in noise_levels:
    print("Noise level", noise_level, "started")
    noise_dir = os.path.join(base_dir, "training", noise_level)
    data_dir = os.path.join(base_dir, "Results", "KAIST Processed", noise_level, "Time")
    ids = []
    labels = {}

    for (
        condition_name
    ) in conditions.keys():  # pylint: disable=consider-using-dict-items
        condition_dir = os.path.join(data_dir, condition_name)

        filenames = os.listdir(condition_dir)
        arr = np.full(np.shape(filenames), condition_name + "_")
        cond_ids = np.char.add(arr, filenames)
        ids.append(cond_ids)

        for cond_id in cond_ids:
            labels[cond_id] = conditions[condition_name]

    ids = np.array(ids).flatten()
    np.random.seed(42)
    np.random.shuffle(ids)

    # loop models
    for m in models.items():
        model_name = m[0]
        print("Model", model_name, "started")
        model_dir = os.path.join(noise_dir, model_name)

        # Model parameters
        input_size = models[model_name]["input_size"]
        n_channels = models[model_name]["n_channels"]
        number_of_blocks = int(models[model_name]["number_of_blocks"])

        # Data parameters
        sensor_index = models[model_name]["sensor_index"]
        datachannel = models[model_name]["datachannel"]

        # Set lists for metrics
        accuracies, f1s, precisions, recalls, training_times = [], [], [], [], []
        split_number = 0

        # Loop splits
        for train, test in kf.split(ids):
            tf.keras.backend.clear_session()

            split_dir = os.path.join(model_dir, "Split{}".format(split_number))
            print("Split", split_number, "started")

            # Data generator
            test_gen = DataGenerator(
                ids[test],
                labels,
                data_dir,
                datachannel,
                n_channels=n_channels,
                sensor_index=sensor_index,
                **params_val
            )
            model = cnn_model(
                input_size, number_of_blocks, n_classes=params["n_classes"]
            )

            model.load_weights(os.path.join(split_dir, "saved_model")).expect_partial()
            true_classes = np.load(os.path.join(split_dir, "true_indices.npy")).astype(
                str
            )
            true_classes = np.char.replace(true_classes, "0", "Normal")
            true_classes = np.char.replace(true_classes, "1", "Outer")
            true_classes = np.char.replace(true_classes, "2", "Inner")
            true_classes = np.char.replace(true_classes, "3", "Ball")

            # Save original model but put another layer as the last one
            model_tsne = Model(
                inputs=[model.input],
                outputs=[model.get_layer(index=layer_index).output],
            )

            # Add flatten layer
            x = Flatten()(model_tsne.output)

            # Save the modified model
            model_tsne = Model(inputs=[model.input], outputs=x)

            # Get features
            try:
                features_tsne = np.load(os.path.join(split_dir, "features_tsne.npy"))
            except FileNotFoundError:
                features_tsne = model_tsne.predict(
                    test_gen, batch_size=params_val["batch_size"]
                )
                np.save(os.path.join(split_dir, "features_tsne.npy"), features_tsne)

            tsne = TSNE(
                n_components=2,
                perplexity=30,
                early_exaggeration=12.0,
                n_iter=1000,
                init="pca",
                learning_rate="auto",
                random_state=42,
            ).fit_transform(features_tsne)
            color_dict = {
                "Normal": "tab:blue",
                "Outer": "tab:orange",
                "Inner": "tab:purple",
                "Ball": "tab:green",
            }
            plot_tsne(split_dir, tsne, true_classes, color_dict=color_dict)

            split_number += 1

            # add break if you only want to create t-sne for the first split
            break
