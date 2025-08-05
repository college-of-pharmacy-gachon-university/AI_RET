# RET Mutation Regression Model

A machine learning pipeline for predicting **pIC50** values of compounds against RET mutations using molecular fingerprints and Random Forest regression.

## Overview

This repository provides a complete workflow for building predictive models for RET mutation activity. The model employs **Extended-Connectivity Fingerprints (ECFP6)** as molecular descriptors and **Random Forest regression** for activity prediction.

> **Note**: The reported model performance is based on pre-defined hyperparameter parameters with a fixed train-test *csv file for reproducibility.

## Repository Structure

```
├── RET_MUT_Regression_Model.ipynb                  # Main training notebook
├── RET_Mutant_SELECTED_TR_TS_threshold_6.0.csv     # Original dataset
├── RET_Mutant_train_set.csv                        # Training split (80%)
├── RET_Mutant_test_set.csv                         # Test split (20%)
├── RET_MUTANT_final_model_Hyper.pkl                # Trained model (generated)
└── README.md                                       # This file
```

## Dataset

The datasets `RET_Mutant_train_set.csv` and `RET_Mutant_test_set.csv` include molecular compounds with corresponding **pIC50** values against RET mutations.  

**Key columns:**
- **RDKIT_SMILES**: Canonical SMILES representation of chemical compounds  
- **pIC50**: Negative logarithm of IC50 values (activity measure)  

The train/test sets are derived from an 80:20 split of the original dataset (`RET_Mutant_SELECTED_TR_TS_threshold_6.0.csv`) using a fixed random seed (**42**) to ensure reproducibility.

### Molecular Descriptors
- **ECFP6 (Extended-Connectivity Fingerprints):**
  - Radius: **3**
  - Features: **On**
  - Counts: **Enabled**
  - Bit length: **2048**

## Model Selection and Hyperparameters

### Final Model Configuration
Hyperparameter optimization was performed using **Optuna** (code excluded). The optimal configuration is:

```python
from sklearn.ensemble import RandomForestRegressor

RandomForestRegressor(
    n_estimators=270,        # Number of trees
    max_depth=28,            # Maximum tree depth
    max_features='auto',     # Features per split
    min_samples_split=2,     # Minimum samples to split
    min_samples_leaf=1,      # Minimum samples per leaf
    bootstrap=True,          # Enable bootstrap sampling
    criterion='mse',         # Split criterion (mean squared error)
    random_state=1234        # Seed for reproducibility
)
```

## Dependencies
```python
# Core libraries
numpy>=1.18.1
pandas>=1.0.3
matplotlib>=3.1.3

# Machine learning
scikit-learn>=0.21.3
scipy>=1.4.1

# Cheminformatics
rdkit>=2020.03.3
```

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd AI_RET/RET_MUT_REG_MODEL
```

2. Install dependencies:
```bash
# Using conda (recommended for RDKit)
conda env create -f env.yml python=3.7.7
conda activate ret-mutation-model

# Or using pip (RDKit installation may require additional steps)
conda env create -n ret-mutation-model python=3.7.7
conda activate ret-mutation-model
pip install -r requirements.txt
# For RDKit installation via pip, see: https://rdkit.readthedocs.io/en/latest/Install.html
```

## Usage

**Run the notebook**
```bash
jupyter notebook RET_MUT_Regression_Model.ipynb
```

## Predicting New Compounds
You can use the trained model (RET_MUTANT_final_model_Hyper.pkl) to predict pIC50 values for new compounds:
```bash
import pickle
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem

# Load trained model
with open('RET_MUTANT_final_model_Hyper.pkl', 'rb') as f:
    model = pickle.load(f)

# Example: Predicting for new SMILES
smiles = ["CCOC1=CC=CC=C1", "CCN(CC)CCOC2=CC=CC=C2"]
def smiles_to_ecfp6(smiles_list):
    return [
        AllChem.GetMorganFingerprintAsBitVect(Chem.MolFromSmiles(s), radius=3, nBits=2048)
        for s in smiles_list
    ]

features = [list(fp) for fp in smiles_to_ecfp6(smiles)]
predictions = model.predict(features)
print("Predicted pIC50 values:", predictions)
```

## Model Performance

The model is evaluated using multiple metrics:

- **R² Score**: Coefficient of determination
- **RMSE**: Root Mean Square Error
- **MAE**: Mean Absolute Error  
- **PCC**: Pearson Correlation Coefficient

Performance is reported separately for:
- Training set
- Test set
- Final model (trained on complete dataset)

## Reproducibility is ensured through:

The pipeline ensures reproducibility through:

- **Fixed random seeds**: SEED=56789 for numpy, random, and environment
- **Deterministic Random Forest**: Random Forest with fixed random_state=1234
- **Pre-defined train/test splits**: Provided CSV files preserve the original partition

## File Descriptions

- **`RET_MUT_Regression_Model_Revised.ipynb`**: End-to-end training workflow notebook
- **`RET_Mutant_SELECTED_TR_TS_threshold_6.0.csv`**: Original dataset with all compounds
- **`RET_Mutant_train_set.csv`**: Training set (80% of data)
- **`RET_Mutant_test_set.csv`**: Test set (20% of data)
- **`RET_MUTANT_final_model_Hyper.pkl`**: Trained Random Forest model (serialized)


## Notes

- Hyperparameter tuning was conducted using Optuna for improved performance.
- The final model was trained on the full dataset for deployment use.
- Predefined train/test splits are retained for fair and reproducible evaluation.
