"""
Author: Tuomas Jalonen

This script is used for training the model and saving results.
"""

# pylint: disable=import-error
# pylint: disable=invalid-name
import os
import json
import tensorflow as tf
from tensorflow.keras.metrics import Precision, Recall
from tensorflow.keras.optimizers.legacy import Adam
from tensorflow.keras.callbacks import LearningRateScheduler
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.model_selection import KFold

from generator import DataGenerator
from utils import (
    cnn_model,
    lr_schedule,
    save_model,
    TimingCallback,
)
from utils import plot_training_curves, plot_cm

# Base_dir is your directory which contains the "Dataset" and "Results" folders
base_dir = "..."

# Define models to be trained and their parameters
models = {
    "cnn2": {
        "input_size": (2000, 2),
        "n_channels": 2,
        "sensor_index": None,
        "number_of_blocks": "4",
        "datachannel": "s",
    }
}

# Define noise levels to be trained
noise_levels = [
    "Clean",
    "-5dB",
    "0dB",
    "5dB",
    "10dB",
    "15dB",
    "20dB",
]
# Define other parameters
params = {
    "dim": (2000,),
    "batch_size": 100,
    "n_classes": 4,
    "shuffle": True,
}

# Validation (testing) parameters are the same but without shuffling
params_val = params
params_val["shuffle"] = False

# Encode labels
conditions = {"Normal": 0, "Outer": 1, "Inner": 2, "Ball": 3}

# Define the number of epochs and cross-validation splits
epochs = 150
kf = KFold(n_splits=5)

# loop noise levels
for noise_level in noise_levels:
    print("Noise level", noise_level, "started")

    # Define paths
    noise_dir = os.path.join(base_dir, "training", noise_level)
    data_dir = os.path.join(base_dir, "Results", "KAIST Processed", noise_level, "Time")

    # Empty list and dictionary for data segment ids and labels
    ids = []
    labels = {}

    # pylint: disable=consider-using-dict-items
    # pylint: disable=consider-iterating-dictionary

    # Loop condition names
    for condition_name in conditions.keys():
        # define path
        condition_dir = os.path.join(data_dir, condition_name)

        # get the filenames of the data segments and add the condition to the file name
        # and append to list
        filenames = os.listdir(condition_dir)
        arr = np.full(np.shape(filenames), condition_name + "_")
        cond_ids = np.char.add(arr, filenames)
        ids.append(cond_ids)

        # Add the correct label to the labels dictionary
        for cond_id in cond_ids:
            labels[cond_id] = conditions[condition_name]

    # Convert ids to numpy array and shuffle
    ids = np.array(ids).flatten()
    np.random.seed(42)
    np.random.shuffle(ids)

    # loop models
    for m in models.items():
        model_name = m[0]
        print("Model", model_name, "started")

        # Model directory
        model_dir = os.path.join(noise_dir, model_name)

        # Model parameters
        input_size = models[model_name]["input_size"]
        n_channels = models[model_name]["n_channels"]
        number_of_blocks = int(models[model_name]["number_of_blocks"])

        # Data parameters
        sensor_index = models[model_name]["sensor_index"]
        datachannel = models[model_name]["datachannel"]

        # Set lists for metrics and start from split 0
        accuracies, f1s, precisions, recalls, training_times = [], [], [], [], []
        split_number = 0

        # Loop splits
        for train, test in kf.split(ids):
            print("Split", split_number, "started")

            # Clear Tensorflow backend to make sure the training starts from scratch
            tf.keras.backend.clear_session()

            # Define split directory
            split_dir = os.path.join(model_dir, f"Split{split_number}")

            # Data generators
            train_gen = DataGenerator(
                ids[train],
                labels,
                data_dir,
                datachannel,
                n_channels=n_channels,
                sensor_index=sensor_index,
                **params,
            )
            test_gen = DataGenerator(
                ids[test],
                labels,
                data_dir,
                datachannel,
                n_channels=n_channels,
                sensor_index=sensor_index,
                **params_val,
            )

            # Get model
            model = cnn_model(
                input_size, number_of_blocks, n_classes=params["n_classes"]
            )

            # Training settings: learning rate scheduling, model saving,
            # measure training time, optimizer
            lr_scheduler = LearningRateScheduler(lr_schedule)
            savemodel = save_model(split_dir)
            timing_callback = TimingCallback()
            opt = Adam(learning_rate=lr_schedule(0), clipnorm=1.0)

            # Compile the model and define loss function and metrics
            model.compile(
                optimizer=opt,
                loss="categorical_crossentropy",
                metrics=[
                    "accuracy",
                    Precision(),
                    Recall(),
                ],
            )

            # Print the model structure
            model.summary()

            # Train the model and save the history
            history = model.fit(
                train_gen,
                validation_data=test_gen,
                epochs=epochs,
                callbacks=[lr_scheduler, savemodel, timing_callback],
            )

            # save the history as numpy and json files
            np.save(os.path.join(split_dir, "history.npy"), history.history)
            hist_df = pd.DataFrame(history.history)
            with open(os.path.join(split_dir, "history.json"), "w") as fp:
                hist_df.to_json(fp)

            # Plot and save training curves
            plot_training_curves(split_dir, history)

            # Load the trained weights and test the model
            model.load_weights(os.path.join(split_dir, "saved_model"))
            predictions = model.predict(test_gen, batch_size=params_val["batch_size"])
            predicted_indices = np.argmax(predictions, axis=-1)
            true_indices = [labels[x] for x in ids[test]]

            # Get confusion matrix and classification report
            cm = confusion_matrix(true_indices, predicted_indices, normalize="true")
            print(cm)
            df_cm = pd.DataFrame(cm)
            plot_cm(split_dir, df_cm)
            cr = classification_report(
                true_indices, predicted_indices, output_dict=True
            )
            print(cr)

            # Save all results as numpy and json files
            np.save(os.path.join(split_dir, "predictions.npy"), predictions)
            with open(os.path.join(split_dir, "predictions.json"), "w") as fp:
                pd.DataFrame(predictions).to_json(fp)

            np.save(os.path.join(split_dir, "predicted_indices.npy"), predicted_indices)
            with open(os.path.join(split_dir, "predicted_indices.json"), "w") as fp:
                pd.DataFrame(predicted_indices).to_json(fp)

            np.save(os.path.join(split_dir, "true_indices.npy"), true_indices)
            with open(os.path.join(split_dir, "true_indices.json"), "w") as fp:
                pd.DataFrame(true_indices).to_json(fp)

            np.save(os.path.join(split_dir, "cm.npy"), cm)
            with open(os.path.join(split_dir, "cm.json"), "w") as fp:
                pd.DataFrame(cm).to_json(fp)

            np.save(os.path.join(split_dir, "cr.npy"), cr)
            with open(os.path.join(split_dir, "cr.json"), "w") as fp:
                pd.DataFrame(cr).to_json(fp)

            # Add split metrics to lists
            accuracies.append(accuracy_score(true_indices, predicted_indices))
            f1s.append(f1_score(true_indices, predicted_indices, average="macro"))
            precisions.append(
                precision_score(true_indices, predicted_indices, average="macro")
            )
            recalls.append(
                recall_score(true_indices, predicted_indices, average="macro")
            )

            # Append training times only for the first split
            if split_number == 0:
                training_times.append(timing_callback.logs)

            # Get misclassification filenames and save them to .npy and .csv
            misclassified_indices = np.where(
                np.not_equal(predicted_indices, true_indices)
            )[0]
            misclassified_ids = ids[test][misclassified_indices]
            np.save(os.path.join(split_dir, "misclassified_ids.npy"), misclassified_ids)
            np.savetxt(
                os.path.join(split_dir, "misclassified_ids.csv"),
                misclassified_ids,
                delimiter="  ",
                comments="",
                fmt="%s",
            )

            # Increment split number
            split_number += 1

        # calculate averages and standard deviations of all metrics and save them
        metrics = [
            {
                "accuracies": accuracies,
                "avg_acc": np.average(accuracies),
                "std_acc": np.std(accuracies),
                "f1s": f1s,
                "avg_f1s": np.average(f1s),
                "std_f1s": np.std(f1s),
                "precisions": precisions,
                "avg_precisions": np.average(precisions),
                "std_precisions": np.std(precisions),
                "recalls": recalls,
                "avg_recalls": np.average(recalls),
                "std_recalls": np.std(recalls),
                "times": training_times,
                "avg_times": np.average(training_times),
                "std_times": np.std(training_times),
            }
        ]

        print(metrics)
        with open(os.path.join(model_dir, "cross-validation_metrics.json"), "w") as fp:
            json.dump(metrics, fp, indent=4)
