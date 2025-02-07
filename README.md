# **BBB Permeability ML Prediction**

**BBB Permeability ML Prediction** is a machine learning-based model designed to predict blood-brain barrier permeability (BBB), yielding a binary **BBB+** or **BBB-** classification.  
This repository provides tools for **data preprocessing, model training, and generating predictions.**  
For **original data**, **preprocessed data** and **data analysis** see the data/B3DB_Processed/ folder.

**Author:** Ben Franey, 2024

---

## **Installation**

### **1. Create a Virtual Environment**
```bash
python3 -m venv venv
```

### **2. Activate the Virtual Environment**
- On **Linux/macOS**:
  ```bash
  source venv/bin/activate
  ```
- On **Windows**:
  ```bash
  venv\Scripts\activate
  ```

### **3. Install Required Packages**
```bash
pip install -r requirements.txt
```

#### **Note:**  
If `RDKit` installation fails, modify the `requirements.txt` entry for RDKit to:
```
rdkit==2024.9.4 --> rdkit
```

---

## **1. Data Preprocessing**
Before making predictions or training a new model, you must preprocess the data.

### **1.1 Data Preparation**
- The input CSV must contain at least the following columns:
  - **'NO.'**: A unique identifier (must be an integer and valid).
  - **'SMILES'**: Molecular representation in SMILES format (must be valid).
  - **'BBB+/BBB-'**: Binary classification for blood-brain barrier permeability (optional, can be empty).
  - **'group'**: Custom grouping information (optional, can be empty).
- Additional columns will be ignored in the final JSON output.
- **PubChemPy runs slowly** due to server restrictions—consider this when enabling PubChem property retrieval - use of PubChem data over RDKit **improves accuracy** in results.

#### **Preprocessing Example Commands**
The following commands preprocess data and output JSON files:

```bash
# Preprocess full B3DB dataset and retrieve PubChem properties
python3 src/preprocessing.py --input_csv data/B3DB_full.csv --output_json data/B3DB_processed/model_ready.json --use_pubchem y

# Preprocess the example dataset for predictions
python3 src/preprocessing.py --input_csv data/example_prediction.csv --output_json data/example_prediction/model_ready.json --use_pubchem y
```

#### **Available Arguments (`argparse`)**
- `--input_csv`: Path to the input CSV file.
- `--output_json`: Path to save the processed JSON file.
- `--use_pubchem`: Whether to retrieve PubChem properties (`y` for yes, `n` for no, default: `n`).
- `--calculate_fragment_properties`: Whether to calculate fragment properties (`y` for yes, `n` for no, default: `n`).

---

### **1.2 Preprocessing Analysis**
Analyze the dataset by generating graphs and statisitcal comparisons, comparing RDKit and PubChem properties, and visualizing property distributions.

#### **Analysis Example Command**
```bash
python3 src/analysis.py --parent_dir data/B3DB_processed
```

#### **Available Arguments (`argparse`)**
- `--parent_dir`: Directory containing the preprocessed JSON and CSV data created through preprocessing for analysis.

---

## **2. Model Training**
Train the model using either **manual parameter settings** or **Optuna-based hyperparameter optimization**.

### **2.1 Training**
Run training with predefined hyperparameters or with Optuna for automatic hyperparameter tuning.

#### **Example Command (No Optuna)**
Use **Defualt Paramaters** for model training.
```bash
python3 src/train.py \
  --data_path data/B3DB_full_model_ready.min.json \
  --output_dir o/output1 \
  --n_folds 10 \
  --random_seed 42 \
  --train_mode 1 \
  --balance_choice 2 \
  --use_gpu n
```

#### **Example Command (Optuna)**
Use **Optuna** for automatic hyperparameter tuning.
```bash
python3 src/train.py \
  --data_path data/B3DB_full_model_ready.min.json \
  --output_dir o/output1-opt10 \
  --n_folds 10 \
  --random_seed 42 \
  --train_mode 2 \
  --balance_choice 2 \
  --use_gpu n \
  --opt_metric all \
  --opt_trials 10
```

#### **Available Arguments (`argparse`)**
- `--data_path`: Path to the preprocessed JSON dataset.
- `--output_dir`: Directory where model outputs will be saved.
- `--n_folds`: Number of cross-validation folds.
- `--random_seed`: Random seed for reproducibility.
- `--balance_choice`: Data balancing strategy (`1` for none, `2` for SMOTE, etc.).
- `--use_gpu`: Use GPU if available (`y` for yes, `n` for no).
- `--train_mode`: Set to `1` for manual training or `2` to enable Optuna optimization.
- `--opt_metric`: Metric to optimize (`auc`, `mcc`, `f1`, etc, or `all`).
- `--opt_trials`: Number of optimization trials to run.

---

## **3. Prediction**
Make predictions on new data, using a trained model.

#### **Example Command**
```bash
python3 src/predict.py \
  --model_path o/output1/best_model.pth \
  --input_json data/example_prediction/model_ready.json \
  --output_csv results/predictions.csv
```

#### **Available Arguments (`argparse`)**
- `--model_path`: Path to the trained model file.
- `--input_json`: Path to the preprocessed input data in JSON format.
- `--output_csv`: Path to save the predictions.

---

## **Directory Structure**
```
BBB-Permeability-ML-Prediction/
|-- data/                      # Input datasets and preprocessed files
|-- data/B3DB_processed/*      # All processed B3DB files, including ananlysis
|-- src/                       # Source code for preprocessing, training, and prediction
|-- published-model/*          # Trained models and related files
|-- predictions/               # Generated predictions
|-- requirements.txt           # List of required Python packages
|-- LICENSE                    # Licence documentation (Apache 2.0 License)
|-- README.md                  # Project documentation (this file)
```

---

## **Script Overview**

### **1. analysis.py**
Performs **data analysis and visualization**, including **comparisons of RDKit vs. PubChem descriptors**, statistical summaries, and property-based histograms.

### **2. decomposition_visualizer.py**
Decomposes **SMILES strings into molecular fragments** (BRICS, RINGS, SIDECHAINS) and saves them as **SVG images** for visualization.

### **3. json_cut.py**
Processes JSON files by **removing unnecessary properties** from **BRICs, RINGS, and SIDE_CHAINS** sections while retaining molecular descriptor information.

### **4. molecule_visualizer.py**
Converts **SMILES strings to 2D molecular structures** and saves them as **SVG images** for visualization.

### **5. predict.py**
Loads a **trained machine learning model** to predict **BBB permeability** from molecular input data and outputs results in a CSV file.

### **6. predictions_check.py**
Compares multiple **prediction outputs**, ensuring consistency across different trained models by analyzing key performance metrics.

### **7. preprocess.py**
Preprocesses molecular data from a CSV file, computing **molecular descriptors and fragment-based properties**, and saves output as JSON and CSV files.

### **8. train.py**
Completes **machine learning model training**, supporting both **manual parameters** and **Optuna-based hyperparameter optimization** for improved model performance.

---

## **License**
This project is open-source under the Apache License, Version 2.0. See `LICENSE` for details.

---

## **Notes**
- Ensure all required dependencies are installed **before running any scripts.**
- If using **GPU acceleration**, make sure CUDA drivers and libraries are correctly installed.
- Ensure all paths are correctly set before running any script.
- For debugging or testing, reduce dataset size or Optuna trial count to speed up execution.
---
