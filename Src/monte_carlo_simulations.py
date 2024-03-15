"""
Author: Tuomas Jalonen

This script is used for doing the Monte Carlo simulations.
"""

# pylint: disable=import-error
# pylint: disable=invalid-name
import os
import time
import tensorflow as tf
import numpy as np
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
from generator import DataGenerator
from utils import cnn_model

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
    # "batch_size": 4,
    "n_classes": 4,
    "shuffle": True,
}
params_val = params
params_val["shuffle"] = False
epochs = 150

# Base_dir is your directory which contains the "data" and "training" folders
base_dir = "..."

noise_levels = [
    # "Clean",
    "-5dB",
    # "0dB",
    # "5dB",
    # "10dB",
    # "15dB",
    # "20dB",
]
conditions = {"Normal": 0, "Outer": 1, "Inner": 2, "Ball": 3}
kf = KFold(n_splits=5)
results = []
# loop noise levels
for noise_level in noise_levels:
    print("Noise level", noise_level, "started")
    noise_dir = os.path.join(base_dir, "training", noise_level)
    data_dir = os.path.join(base_dir, "Results", "KAIST Processed", noise_level, "Time")
    ids = []
    labels = {}
    # pylint: disable=consider-using-dict-items
    # pylint: disable=consider-iterating-dictionary
    for condition_name in conditions.keys():
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

            split_dir = os.path.join(model_dir, f"Split{split_number}")
            print("Split", split_number, "started")

            # Data generators
            test_gen = DataGenerator(
                ids[test],
                labels,
                data_dir,
                datachannel,
                n_channels=n_channels,
                sensor_index=sensor_index,
                **params_val,
            )
            model = cnn_model(
                input_size, number_of_blocks, n_classes=params["n_classes"]
            )

            model.load_weights(os.path.join(split_dir, "saved_model")).expect_partial()
            times = np.empty(10000, dtype=float)
            index = 0
            for i in range(1000):
                if i % 100 == 0:
                    print(i)
                sample = test_gen[i][0][0]
                sample = np.expand_dims(sample, axis=0)

                for j in range(10):
                    start_time = time.time()
                    prediction = model.predict(sample, verbose=0)
                    times[index] = time.time() - start_time
                    index += 1
            results.append(
                (
                    "model name:",
                    model_name,
                    "number of predictions:",
                    len(times),
                    "mean prediction time:",
                    np.mean(times),
                    "std prediction time:",
                    np.std(times),
                )
            )
            print(results)
            np.savetxt(
                os.path.join(split_dir, "monte_carlo_results.csv"),
                results,
                delimiter=",",
                comments="",
                fmt="%s",
            )
            np.savetxt(
                os.path.join(split_dir, "monte_carlo_times.csv"),
                times,
                delimiter=",",
                comments="",
                fmt="%s",
            )
            plt.plot(times)
            plt.savefig(os.path.join(split_dir, "monte_carlo_times.pdf"), dpi=300)

            split_number += 1
            break
    break
