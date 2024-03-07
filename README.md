# Real-Time Vibration-Based Bearing Fault Diagnosis Under Time-Varying Speed Conditions
An efficient real-time CNN for diagnosing multiple bearing faults under various noise levels and time-varying rotational speeds with a Fisher-based spectral separability analysis method to elucidate its effectiveness.

The material in this repository is provided to supplement the following paper: Jalonen, T., Al-Sa'd, M., Kiranyaz, S., & Gabbouj, M. (2024). Real-Time Vibration-Based Bearing Fault Diagnosis Under Time-Varying Speed Conditions. In 25th IEEE International Conference on Industrial Technology.

The MATLAB and Python scripts and data listed in this repository are used to produce results, and supporting figures illustrated in the paper.

## Data:
The repository contains data within each of these folders:
### Dataset:
-   This folder holds the raw KAIST dataset. For copyright reasons, please download and extract the KAIST dataset files from here:
-   Part 1: https://data.mendeley.com/datasets/vxkj334rzv/7
-   Part 2: https://data.mendeley.com/datasets/x3vhp8t6hg/7
-   Part 3: https://data.mendeley.com/datasets/j8d8pfkvj2/7
### Results\KAIST Processed:
-   This folder holds the output of the MATLAB script *Main_preprocessing.m*; the preprocessed vibration segments for each SNR level and fault class.
### Results\Computational Complexity:
-   It contains the output of the MATLAB script *Demo_3_computational_complexity.m*; the complexity analysis results for the proposed model.
### Results\Performance:
-   This folder contains the performance results of the proposed model for each cross-validation fold

## MATLAB Scripts:
The repository contains the following MATLAB scripts within its directory:
### Main_preprocessing.m
-   This main script pre-processes the vibration signals in the KAIST dataset and saves the processed segments.
### Demo_1_speed_spectra.m
-   This demo script produces the frequency analysis results in Fig. 2.
### Demo_3_computational_complexity.m
-   This script generates the computational complexity analysis results in Fig. 6.
### Demo_4_spectral_separability_analysis.m
-   This demo script performs the Fisher-based spectral separability analysis and generates the results presented in Fig. 7.

## Python Scripts:
### xxx
-   xxx
### yyy
-   yyy
### zzz
-   zzz
