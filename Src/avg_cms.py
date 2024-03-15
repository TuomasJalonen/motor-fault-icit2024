"""
Author: Tuomas Jalonen

This script is used for plotting the average confusion matrices of the 5-fold cross-validation.
"""

import os
import numpy as np
import pandas as pd
from utils import plot_cm

# Base_dir is your directory which contains the "Dataset" and "Results" folders
base_dir = "..."

noise_levels = [
    "Clean",
    "-5dB",
    "0dB",
    "5dB",
    "10dB",
    "15dB",
    "20dB",
]

# loop noise levels
for noise_level in noise_levels:
    noise_dir = os.path.join(base_dir, "training", noise_level)
    model_dir = os.path.join(noise_dir, "CNN2")
    cms = []
    for split_number in range(5):
        split_dir = os.path.join(model_dir, "Split{}".format(split_number))
        cm = np.load(os.path.join(split_dir, "cm.npy"))
        cms.append(cm)
    avg_cm = np.mean(cms, axis=0)
    avg_cm = np.around(avg_cm, decimals=3)
    avg_cm = pd.DataFrame(
        avg_cm,
        index=[i for i in ["Normal", "Outer", "Inner", "Ball"]],
        columns=[i for i in ["Normal", "Outer", "Inner", "Ball"]],
    )

    plot_cm(model_dir, avg_cm)
