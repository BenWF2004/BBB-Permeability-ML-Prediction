# **BBB Permeability ML Prediction**

**BBB Permeability ML Prediction** is a machine learning-based model designed to predict blood-brain barrier permeability (BBB), yielding a binary **BBB+** or **BBB-** classification.  
This repository provides tools for **data preprocessing, model training, and generating predictions.**  

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

## **Usage**

### **1. Data Preprocessing**
To **make predictions** or **train your own model**, first preprocess the data:
The inpuput CSV must have the rows: 'NO.', 'SMILES', 'BBB+/BBB-', and 'group'. Rows 'NO.' and 'SMILES' must be valid, but 'BBB+/BBB-' and 'group' can be empty.

```bash
python3 src/preprocessing.py --input_csv data/B3DB_full.csv --output_json data/B3DB_full_model_ready.json
```
A **pretrained model (optimized through 75 trials)** is provided in the `published-model/` directory to run predictions on..

### **1.2 Preprocessing Analysis**
Analyze the dataset through **graph generation, RDKit vs PubChem comparisons, and property distributions**:

```bash
python3 src/analysis.py --parent_dir data/B3DB_processed
```

---

### **2. Model Training**

#### **2.1 Training Without Optuna (Manual Parameters)**
Train the model using predefined parameters (no hyperparameter tuning):

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

#### **2.2 Training With Optuna (Hyperparameter Optimization)**
Use **Optuna** for automatic hyperparameter optimization:

```bash
python3 src/train.py \
  --data_path data/B3DB_full_model_ready.min.json \
  --output_dir o/output1 \
  --n_folds 10 \
  --random_seed 42 \
  --train_mode 2 \
  --balance_choice 2 \
  --use_gpu n \
  --opt_metric all \
  --opt_trials 2
```

---

### **3. Prediction**
Use a **trained model** to generate predictions.
Run the **Data Preprocessing** script on the prediction data to generate the input JSON.

```bash
python3 src/predict.py \
  --model_dir published-model \
  --input_json published-model/validation_data_original.json \
  --output_csv predictions/validation_data_original.csv
```

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
|-- LICENSE                    # Licence documentation (MIT License)
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

## **Notes**
- Ensure all required dependencies are installed **before running any scripts.**
- If using **GPU acceleration**, make sure CUDA drivers and libraries are correctly installed.

---
