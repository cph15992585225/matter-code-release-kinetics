# Source Code for Matter Submission

This repository contains the Python source code associated with a manuscript submitted to *Matter*.  
It is provided to support code evaluation and reproducibility assessment during peer review.

## Reviewer Instructions

The scripts are configured to read the input Excel files from the `datas/` directory.

The files `training_data.xlsx` and `prediction_data.xlsx` are provided as attachments in the journal submission system for peer review. To run the code successfully, reviewers should download these files and manually place them into the `datas/` folder under the repository root before executing the scripts.


## Repository Contents

- `dataset.py`  
  Utility functions for reading Excel datasets, extracting features and labels, and preparing data loaders.

- `ML_for_opt_ok_matter.py`  
  Machine learning workflow for regression modeling, Bayesian hyperparameter optimization, model evaluation, feature importance analysis, and SHAP-based interpretation.

- `DL_normal_bayes_OK_matter.py`  
  Deep learning workflow for regression modeling, hyperparameter optimization, model evaluation, and prediction on an external dataset.

## Computational Environment

All computational workflows were implemented in Python 3.8.

The main libraries and versions used in this work are:

- scikit-learn 1.3.2
- XGBoost 2.1.4
- LightGBM 4.5.0
- PyTorch 1.12.1
- Optuna 4.2.1
- SHAP 0.44.1
- pandas 1.5.3
- NumPy 1.24.3
- Matplotlib 3.7.5
- Seaborn 0.13.2

Additional dependencies required for running the code include:

- scipy
- scikit-optimize
- openpyxl

