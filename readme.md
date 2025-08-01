# Overview

## Reinforcement Learning-Based RET-Specific Molecular Design

This repository hosts the code, data, and documentation for the project:

#### Generative Model-Guided Discovery of Amino Quinoxaline Scaffold as a Novel RET Alteration Inhibitor with Exploration of the Cryptic Pocket

Authors:
Surendra Kumar, Vinay Pogaku, Mi-Hyun Kim*

Affiliation:
Gachon Institute of Pharmaceutical Science and Department of Pharmacy, College of Pharmacy, Gachon University, 191 Hambakmoe-ro, Yeonsu-gu, Incheon, Republic of Korea

*Corresponding author: kmh0515@gachon.ac.kr

## Requirements and Installation
This project required several external tools and software modules. Please ensure the following are installed on your system for successful execution:

1. REINVENT v3.2

2. KNIME Analytics Platform

3. [Schrödinger Software Suite (2020-4 or later)] — Requires a valid license

4. [OpenEye Scientific Software] — Requires a valid license

## Dataset Collection
To facilitate RET-specific data curation, we provide a pre-built KNIME workflow (.knwf file) in `KNIME_WORKFLOW folder`:

#### File: `RET_DATASET.knwf`

#### How to use the KNIME workflow:
Launch the KNIME Analytics Platform.

Navigate to: File > Import KNIME Workflow.

Select and import the provided .knwf file.

Execute the workflow within the KNIME workspace to collect and preprocess the RET-specific dataset.

## Molecular Generation using Reinforcement Learning
This repository includes multiple .json configuration files under `RL_JSON_FILE folder` for executing reinforcement learning (RL) jobs using REINVENT v3.2.

#### Setup:
To install REINVENT v3.2, please follow the instructions on the official REINVENT GitHub repository (https://github.com/MolecularAI/Reinvent).

Once installed, you may run individual jobs as follows:

```bash
cd RL_JSON_FILE/Job01
python input.py RL_config.json
```
⚠️ Note: Before running any job, make sure to update the file paths in the .json configuration files for reading input and writing output.

## RET Regression Model (G810R Mutant)
This folder contains the dataset for the RET G810R mutant, along with a Jupyter Notebook used for training, prediction, and saving the regression model. The trained model used in this work is also included.

## Post-Processing and Analysis
A KNIME workflow `(AI_RET_MANUSCRIPT_DATA.knwf)` is provided for post-processing under `KNIME_WORKFLOW folder` the generated molecules and analyzing the outputs from the reinforcement learning (RL) jobs. Please note that the exact procedure used in this study is implemented in the workflow. However, due to confidentiality, the SMILES structures of all the de novo generated molecules are not publicly disclosed or included in this repository.

## Inception File
An inception file containing known RET-specific ligands is included to guide the reinforcement learning model during training. This file can be directly used in the configuration files `*json` of REINVENT.

# Additional Information:
For any queries mail to Mi-hyun Kim (kmh0515@gachon.ac.kr) or Surendra Kumar (surendramph@gmail.com)






