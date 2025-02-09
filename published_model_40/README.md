# 40 Optuna Trials Model

## Overview
This repository contains a machine learning model trained using **40 Optuna trials** for hyperparameter optimization. The model was originally developed using **cross-validation (CV) averaging** but has since been updated to evaluate performance based on an **unseen hold-out set**.

## Key Details
- This model is **old** and has undergone significant updates in the new models.
- Unlike the latest version, this model does **not** incorporate the optimized **BRIC, SIDE CHAINS, and RINGS** training methodology - making it incompatable with the current prediction script.
- It follows a **generic 5-5-5 format**, whereas the current framework integrates these features into a singular model.
- The current **prediction script is incompatible** with this model due to these structural differences.

## Model Training
- **Optimization**: Conducted over **40 trials** using **Optuna**.
- **Evaluation**: Based on **CV average**.
- **Feature Representation**: Uses a **5-5-5 architecture**, differing from the newer version which optimizes **BRIC, SIDE CHAINS, and RINGS** within a unified model.

## Compatibility & Usage
**This model is not compatible with the current prediction script.**


## Future Considerations
For best results, use to the latest version that integrates the new feature optimization approach. If legacy compatibility is required, adjustments to the prediction script will be necessary.
This model exists **soley as legacy** for the webapp untill the update can be testes and moved over.

---
**Author:** Ben Franey
**Last Updated:** 09/02/2025