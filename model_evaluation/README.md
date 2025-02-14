# Model Evaluation: Differing Data Sources and Balancing

## Overview
This directory contains **machine learning models** trained using various **data sources** (RDKit, PubChem, Average, Both, and PubChem else RDKit) and **data balancing techniques** (None, SMOTE, SMOTEENN, SMOTETomek). It also includes predictions and performance assessments to determine the optimal model for publication.

## Contents
- **Trained Models**: Singular models trained on different combinations of data sources and balancing techniques.
- **Predictions**: Output predictions from each trained model.
- **Performance Assessments**: Evaluation metrics (AUC, MCC, F1, Sensitivity, Specificity) to compare models.

## Folder Structure
```
model_comparisons_smote_rdkit_pubchem/
|-- models/                     # Trained models
|-- predictions/                # Prediction outputs from each model
|-- model_evaluations.csv       # Raw performance metrics
|-- model_evaluations.xlsx      # Performance metrics and analysis
|-- README.md                   # Evaluation documentation (this file)
|-- run-predict-param.sh        # Sh file to run model prediction*
|-- run-train-param.sh          # Sh file to run the model training*
```

```*``` Note: these files should not be run as-is, each section required train.py and predict.py to be modified to manually change data sources.

## Usage
- **Models**: Trained using `train.py` with appropriate flags for data source and balancing method.
- **Predictions**: Generated using `predict.py` from trained models.
- **model_evaluations.csv**: Contains performance metrics for model selection.

## Data Sources
- **RDKit**: Calculated molecular descriptors.
- **PubChem**: Retrieved experimental data.
- **Average**: Averaged RDKit and PubChem data.
- **Both**: Seperate entries for both RDKit and PubChem.
- **PubChem else RDKit**: Uses PubChem data if available, else defaults to RDKit.

## Balancing Techniques
- **None**: No data balancing.
- **SMOTE**: Synthetic Minority Over-sampling Technique.
- **SMOTEENN**: Combination of SMOTE and Edited Nearest Neighbors.
- **SMOTETomek**: Combination of SMOTE and Tomek links.

## How to Run
This is a pure post-run analytics section. The .sh files should not be run.

## Training Hyperparameters Used
```
PREDEFINED_PARAMS = {
    'max_depth': 10,
    'learning_rate': 0.016303454,
    'subsample': 0.682094104,
    'colsample_bytree': 0.667401171,
    'min_child_weight': 1,
    'gamma': 0.257201234,
    'alpha': 0.010582666,
    'lambda': 1.204193852
}
```

## Author
**Ben Franey**  
Version: 1.0  
Last Updated: 14/02/2025