# Real-Time Vibration-Based Bearing Fault Diagnosis Under Time-Varying Speed Conditions
An efficient real-time CNN for diagnosing multiple bearing faults under various noise levels and time-varying rotational speeds with a Fisher-based spectral separability analysis method to elucidate its effectiveness.

The material in this repository is provided to supplement the following paper:
Jalonen, T., Al-Sa'd, M., Kiranyaz, S., & Gabbouj, M. (2024). Real-Time Vibration-Based Bearing Fault Diagnosis Under Time-Varying Speed Conditions. In 25th IEEE International Conference on Industrial Technology. https://doi.org/10.48550/arXiv.2311.18547

The MATLAB and Python scripts and data listed in this repository are used to produce results, and supporting figures illustrated in the paper.

## Data
The repository contains data within each of these folders:
### Dataset
-   This folder holds the raw KAIST dataset. For copyright reasons, please download and extract the KAIST dataset files from here:
-   Part 1: https://data.mendeley.com/datasets/vxkj334rzv/7
-   Part 2: https://data.mendeley.com/datasets/x3vhp8t6hg/7
-   Part 3: https://data.mendeley.com/datasets/j8d8pfkvj2/7
### Results\KAIST Processed
-   This folder holds the output of the MATLAB script *Main_preprocessing.m*; the preprocessed vibration segments for each SNR level and fault class.

## MATLAB Scripts
The repository contains the following MATLAB scripts within its directory:
### Main_preprocessing.m
-   This main script pre-processes the vibration signals in the KAIST dataset and saves the processed segments.
### Demo_1_speed_spectra.m
-   This demo script produces the frequency analysis results in Fig. 2.
### Demo_3_computational_complexity.m
-   This script generates the computational complexity analysis results in Fig. 6.
### Demo_4_spectral_separability_analysis.m
-   This demo script performs the Fisher-based spectral separability analysis and generates the results presented in Fig. 7.

## Python Scripts
### train.py
-   This script is used for training the model and saving results.
### utils.py
-   This file contains utility functions.
### generator.py
-   This file contains a custom Keras generator, which reads .mat-files (Matlab).
### t-sne.py
-   This script is used for creating the t-sne results.
### avg_cms.py
-   This script is used for plotting the average confusion matrices of the 5-fold cross-validation.
### monte_carlo_simulations.py
-   This script is used for doing the Monte Carlo simulations.
