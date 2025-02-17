"""
Author: Ben Franey
Version: 11.2.3 - Publish: 1.2
Last Review Date: 06-02-2025
Overview:
This script implements a complete pipeline for training and optimizing an XGBoost model 
to classify molecules as BBB+ (blood-brain barrier permeable) or BBB- (non-permeable). 
It integrates feature engineering, data preprocessing, hyperparameter tuning, 
and model evaluation with various performance metrics.

Key Features:
Data Preprocessing:
  - Loads molecular data from a JSON file.
  - Extracts key molecular descriptors and fragments (BRICS, RINGS, SIDE_CHAINS).
  - Cleans missing values and standardizes classification labels.

Feature Engineering:
  - Splits data into train/validation first, ensuring no data leakage.
  - Builds fragment dictionaries/frequencies only from the training set.
  - Applies the same dictionary to transform the validation set (no frequency counting on val).
  - Handles missing values using KNN imputation.
  - Supports multiple data balancing methods (None, SMOTE, SMOTEENN, SMOTETomek).

Training and Optimization:
  - Trains an XGBoost model with predefined hyperparameters or via Optuna-based hyperparameter optimization.
  - Uses cross-validation on the training portion to log metrics (e.g. recall).
  - Uses the hold-out validation set for final evaluation or Optuna objective.
  - Preforms optimization to metrics specifically from final hold-out set not CV averge.
  - Ensures no data leakage between folds or from validation.

Model Evaluation:
  - The hold-out validation set remains the same for all trials (no re-splitting).
  - Generates ROC curves, calibration curves, and confusion matrices.
  - Computes SHAP (SHapley Additive exPlanations) values for feature importance analysis.
  - Produces feature correlation heatmaps and importance rankings.

Results and Output:
  - Saves trained models and Optuna optimization logs.
  - Outputs validation set performance metrics.
  - For each final model (default or best-Optuna), saves fragment vectorizers,
    kept-tokens joblibs, CSV matrices, confusion matrix, SHAP plots, etc.
  - Produces a comparison chart (CV average vs final evaluation) for all trials in Optimize mode.
  
Usage example:
python3 src/train.py \
  --data_path data/B3DB_processed/processed.cut.min.json \
  --output_dir o/output1-opt10 \
  --n_folds 10 \
  --random_seed 42 \
  --train_mode 2 \
  --balance_choice 1 \
  --use_gpu n \
  --opt_metric all \
  --opt_trials 10
  
Available Arguments (argparse)
--data_path: Path to the preprocessed JSON dataset.
--output_dir: Directory where model outputs will be saved.
--n_folds: Number of cross-validation folds.
--random_seed: Random seed for reproducibility.
--balance_choice: Data balancing strategy (1 for none, 2 for SMOTE, etc.).
--use_gpu: Use GPU if available (y for yes, n for no).
--train_mode: Set to 1 for manual training or 2 to enable Optuna optimization.
--opt_metric: Metric to optimize (auc, mcc, f1, etc, or all).
--opt_trials: Number of optimization trials to run.

"""

import os
import json
import sys
import logging
import warnings
import joblib
import numpy as np
import pandas as pd
from tqdm import tqdm
import argparse
import re

from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import (
    roc_auc_score, confusion_matrix, f1_score, roc_curve,
    matthews_corrcoef, balanced_accuracy_score, precision_score,
    accuracy_score, recall_score
)
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.calibration import calibration_curve
from sklearn.impute import KNNImputer

from imblearn.over_sampling import SMOTE
from imblearn.combine import SMOTEENN, SMOTETomek

import xgboost as xgb
import shap
import matplotlib.pyplot as plt
import seaborn as sns
import optuna

DEFAULT_N_FOLDS = 10
DEFAULT_RANDOM_SEED = 42
MINIMUM_FRAGMENT_FREQUENCY = [3, 3, 3, 5]  # [default, BRICS, RINGS, SIDECHAINS]

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

warnings.filterwarnings("ignore")


def setup_logging(log_file_path):
    """
    Configures logging with both file and console handlers.
    - Saves all logs to a specified file.
    - Outputs warnings and errors to the console.
    - Uses timestamped, severity-labeled log entries.
    
    Parameters:
        log_file_path (str): Path to the log file where logs will be recorded.
    
    Returns:
        None: Initializes logging but does not return any values.
    """
    
    logging.basicConfig(
        filename=log_file_path,
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    console = logging.StreamHandler()
    console.setLevel(logging.WARNING)
    formatter = logging.Formatter('%(levelname)s - %(message)s')
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)


def sanitize_column_names(df):
    """
    Cleans column names by replacing special characters that may cause issues in XGBoost.

    - Identifies and replaces problematic characters '[', ']', '<', '>' with '_' - primairy from BRICS
    - Ensures column names are compatible with XGBoost training.
    - Modifies column names in place without altering data structure.

    Parameters:
        df (pd.DataFrame): DataFrame with potentially problematic column names.

    Returns:
        pd.DataFrame: DataFrame with sanitized column names.
    """
    
    new_cols = []
    # Iterate though each colum
    for c in df.columns:
        c_str = str(c)
        
        # Check for bad characters
        for bad in ['[', ']', '<', '>']:
            c_str = c_str.replace(bad, '_')
        new_cols.append(c_str)
    
    # Replace df with new colums
    df.columns = new_cols
    return df


def revert_brics_naming(column_name):
    """
    Restores BRICS fragment notation by converting specific underscore-separated patterns.

    - Detects patterns like '_14*_' and converts them to '[14*]'.
    - Applies transformation only when format matches '_digit(s)*_'.
    - Uses regular expressions to perform safe substitutions.

    Parameters:
        column_name (str): Column name containing BRICS fragment notation.

    Returns:
        str: Reformatted column name with BRICS notation restored.
    """
    
    return re.sub(r'_(\d+)\*_', r'[\1*]', column_name)


def load_data(data_path):
    """
    Loads molecular data from a JSON file.

    - Reads a JSON file from the specified path.
    - Returns the data as a list of dictionary objects.
    - Logs successful loading or captures errors.
    - Exits the program if loading fails.

    Parameters:
        data_path (str): Path to the JSON file.

    Returns:
        list: Parsed JSON content as a list of dictionaries.

    Raises:
        SystemExit: If the file cannot be loaded.
    """
    
    try:
        with open(data_path, 'r') as f:
            data = json.load(f)
        logging.info(f"Data loaded from {data_path}")
        return data
    except Exception as e:
        logging.error(f"Error loading data: {e}")
        sys.exit(f"Critical error: {e}")


def preprocess_data(data):
    """
    Converts JSON molecular data into a structured DataFrame.

    - Filters out entries missing essential classification data.
    - Extracts molecule-level properties, including molecular weight, LogP, and charge.
    - Selects between RDKit and PubChem descriptors, prioritizing PubChem when available.
    - Extracts molecular fragments (BRICs, RINGS, SIDE_CHAINS) and stores them as space-separated SMILES strings.
    - Returns a processed DataFrame for downstream analysis.

    Parameters:
        data (list): List of molecular entries in dictionary format.

    Returns:
        pd.DataFrame: Processed DataFrame containing molecular descriptors and fragment data.
    """
    
    df_list = []
    
    # initialize tqdm bar for each molecule.
    for entry in tqdm(data, desc="Processing entries"):
        
        # Skip incorrect entries
        if '_preface' in entry:
            continue
        if 'BBB+/BBB-' not in entry:
            continue
        
        """ Extracted from row_data as causing errors"""
        # Testing: Use PubChem only if exist then RDKit if not
        """# Testing: 'LogP': entry.get('LogP_PubChem', np.nan) 
            if not pd.isna(entry.get('LogP_PubChem', np.nan)) 
            else entry.get('LogP_RDKit', np.nan),
        'Flexibility': entry.get('Flexibility_PubChem', np.nan) 
            if not pd.isna(entry.get('Flexibility_PubChem', np.nan)) 
            else entry.get('Flexibility_RDKit', np.nan),
        'HBA': entry.get('HBA_PubChem', np.nan) 
            if not pd.isna(entry.get('HBA_PubChem', np.nan)) 
            else entry.get('HBA_RDKit', np.nan),
        'HBD': entry.get('HBD_PubChem', np.nan) 
            if not pd.isna(entry.get('HBD_PubChem', np.nan)) 
            else entry.get('HBD_RDKit', np.nan),
        'TPSA': entry.get('TPSA_PubChem', np.nan) 
            if not pd.isna(entry.get('TPSA_PubChem', np.nan)) 
            else entry.get('TPSA_RDKit', np.nan),
        'Charge': entry.get('Charge_PubChem', np.nan) 
            if not pd.isna(entry.get('Charge_PubChem', np.nan)) 
            else entry.get('Charge_RDKit', np.nan),
        'Atom_Stereo': entry.get('AtomStereo_PubChem', np.nan) 
            if not pd.isna(entry.get('AtomStereo_PubChem', np.nan)) 
            else entry.get('AtomStereo_RDKit', np.nan),"""
            
        # Testing: PubChem only
        """'LogP': entry.get('LogP_PubChem', np.nan),
        'Flexibility': entry.get('Flexibility_PubChem', np.nan),
        'HBA': entry.get('HBA_PubChem', np.nan),
        'HBD': entry.get('HBD_PubChem', np.nan),
        'TPSA': entry.get('TPSA_PubChem', np.nan),
        'Charge': entry.get('Charge_PubChem', np.nan),
        'Atom_Stereo': entry.get('AtomStereo_PubChem', np.nan),"""
        
        # Testing: Average (RDKit and PubChem) only
        """'LogP': entry.get('LogP_Avg', np.nan),
        'Flexibility': entry.get('Flexibility_Avg', np.nan),
        'HBA': entry.get('HBA_Avg', np.nan),
        'HBD': entry.get('HBD_Avg', np.nan),
        'TPSA': entry.get('TPSA_Avg', np.nan),
        'Charge': entry.get('Charge_Avg', np.nan),
        'Atom_Stereo': entry.get('AtomStereo_Avg', np.nan),"""
        
        # Testing: Both RDKit and PubChem
        """'LogP-P': entry.get('LogP_PubChem', np.nan),
        'Flexibility-P': entry.get('Flexibility_PubChem', np.nan),
        'HBA-P': entry.get('HBA_PubChem', np.nan),
        'HBD-P': entry.get('HBD_PubChem', np.nan),
        'TPSA-P': entry.get('TPSA_PubChem', np.nan),
        'Charge-P': entry.get('Charge_PubChem', np.nan),
        'Atom_Stereo-P': entry.get('AtomStereo_PubChem', np.nan),
        
        'LogP-R': entry.get('LogP_RDKit', np.nan),
        'Flexibility-R': entry.get('Flexibility_RDKit', np.nan),
        'HBA-R': entry.get('HBA_RDKit', np.nan),
        'HBD-R': entry.get('HBD_RDKit', np.nan),
        'TPSA-R': entry.get('TPSA_RDKit', np.nan),
        'Charge-R': entry.get('Charge_RDKit', np.nan),
        'Atom_Stereo-R': entry.get('AtomStereo_RDKit', np.nan),"""

        # Add row data for each molecule
        row_data = {
            'NO.': entry.get('NO.', np.nan),
            'BBB': entry.get('BBB+/BBB-', np.nan),
            'SMILES': entry.get('SMILES', np.nan),
            
            # Testing: RDKit only
            'LogP': entry.get('LogP_RDKit', np.nan),
            'Flexibility': entry.get('Flexibility_RDKit', np.nan),
            'HBA': entry.get('HBA_RDKit', np.nan),
            'HBD': entry.get('HBD_RDKit', np.nan),
            'TPSA': entry.get('TPSA_RDKit', np.nan),
            'Charge': entry.get('Charge_RDKit', np.nan),
            'Atom_Stereo': entry.get('AtomStereo_RDKit', np.nan),

            # More Descriptors
            'Molecular_Weight': entry.get('MW_RDKit', np.nan),
            'Heavy_Atom_Count': entry.get('HeavyAtom_RDKit', np.nan),
            'Radius_Of_Gyration': entry.get('RadiusOfGyration_RDKit', np.nan),
            'Wiener_Index': entry.get('WienerIndex_RDKit', np.nan),
            'Eccentric_Connectivity_Index': entry.get('EccentricConnectivityIndex_RDKit', np.nan),

            # Atom Counts
            'Atom_Count_C': entry.get('AtomCount_C_RDKit', np.nan),
            'Atom_Count_H': entry.get('AtomCount_H_RDKit', np.nan),
            'Atom_Count_O': entry.get('AtomCount_O_RDKit', np.nan),
            'Atom_Count_N': entry.get('AtomCount_N_RDKit', np.nan),
            'Atom_Count_S': entry.get('AtomCount_S_RDKit', np.nan),
            'Atom_Count_F': entry.get('AtomCount_F_RDKit', np.nan),
            'Atom_Count_Cl': entry.get('AtomCount_Cl_RDKit', np.nan),
            'Atom_Count_Br': entry.get('AtomCount_Br_RDKit', np.nan),
            'Atom_Count_I': entry.get('AtomCount_I_RDKit', np.nan),
            'Atom_Count_P': entry.get('AtomCount_P_RDKit', np.nan),
            'Atom_Count_B': entry.get('AtomCount_B_RDKit', np.nan),
            'Atom_Count_Li': entry.get('AtomCount_Li_RDKit', np.nan),

            # Bond Counts
            'Bond_Count_Single': entry.get('BondCount_Single_RDKit', np.nan),
            'Bond_Count_Double': entry.get('BondCount_Double_RDKit', np.nan),
            'Bond_Count_Triple': entry.get('BondCount_Triple_RDKit', np.nan),
            'Bond_Count_Aromatic': entry.get('BondCount_Aromatic_RDKit', np.nan),

            # Chiral Centers
            'Total_Chiral_Centers': entry.get('Total_Chiral_Centers_RDKit', np.nan),
            'R_Isomers': entry.get('R_Isomers_RDKit', np.nan),
            'S_Isomers': entry.get('S_Isomers_RDKit', np.nan),

            # Geometric Isomers
            'E_Isomers': entry.get('E_Isomers_RDKit', np.nan),
            'Z_Isomers': entry.get('Z_Isomers_RDKit', np.nan),

            # Ring Descriptors
            'Num_4_Rings_Aromatic': entry.get('Num_4_Rings_Aromatic_RDKit', np.nan),
            'Num_4_Rings_NonAromatic': entry.get('Num_4_Rings_NonAromatic_RDKit', np.nan),
            'Num_4_Rings_Total': entry.get('Num_4_Rings_Total_RDKit', np.nan),
            'Num_5_Rings_Aromatic': entry.get('Num_5_Rings_Aromatic_RDKit', np.nan),
            'Num_5_Rings_NonAromatic': entry.get('Num_5_Rings_NonAromatic_RDKit', np.nan),
            'Num_5_Rings_Total': entry.get('Num_5_Rings_Total_RDKit', np.nan),
            'Num_6_Rings_Aromatic': entry.get('Num_6_Rings_Aromatic_RDKit', np.nan),
            'Num_6_Rings_NonAromatic': entry.get('Num_6_Rings_NonAromatic_RDKit', np.nan),
            'Num_6_Rings_Total': entry.get('Num_6_Rings_Total_RDKit', np.nan),
            'Num_8_Rings_Aromatic': entry.get('Num_8_Rings_Aromatic_RDKit', np.nan),
            'Num_8_Rings_NonAromatic': entry.get('Num_8_Rings_NonAromatic_RDKit', np.nan),
            'Num_8_Rings_Total': entry.get('Num_8_Rings_Total_RDKit', np.nan),
            'Num_Aromatic_Rings': entry.get('Num_Aromatic_Rings_RDKit', np.nan),
            'Num_NonAromatic_Rings': entry.get('Num_NonAromatic_Rings_RDKit', np.nan),
            'Num_Total_Rings': entry.get('Num_Total_Rings_RDKit', np.nan)
        }

        # Collect BRICS
        brics = entry.get('BRICs', [])
        bric_smiles_list = [b.get('BRIC', '').strip() for b in brics if b.get('BRIC', '').strip()]
        row_data['BRIC_SMILES'] = ' '.join(bric_smiles_list)

        # Collect RINGS
        rings = entry.get('RINGS', [])
        ring_smiles_list = [r.get('RING', '').strip() for r in rings if r.get('RING', '').strip()]
        row_data['RINGS_SMILES'] = ' '.join(ring_smiles_list)

        # Collect SIDE_CHAINS
        side_chains = entry.get('SIDE_CHAINS', [])
        side_chain_smiles_list = [sc.get('SIDE_CHAIN', '').strip() for sc in side_chains if sc.get('SIDE_CHAIN', '').strip()]
        row_data['SIDE_CHAINS_SMILES'] = ' '.join(side_chain_smiles_list)

        df_list.append(row_data)

    # Convert processed data into a structured DataFrame
    df = pd.DataFrame(df_list)
    logging.info(f"Preprocessed DataFrame shape: {df.shape}")
    return df


def handle_missing_values(df):
    """
    Handles missing values in the dataset.

    - Ensures the 'BBB' classification column exists, otherwise exits with an error.
    - Drops rows where the 'BBB' column is missing, as classification is essential.
    - Fills missing values in fragment-related text columns ('BRIC_SMILES', 'RINGS', 'SIDE_CHAINS') with empty strings.

    Parameters:
        df (pd.DataFrame): Input DataFrame containing molecular data.

    Returns:
        pd.DataFrame: Processed DataFrame with missing values handled.

    Raises:
        SystemExit: If the 'BBB' column is missing.
    """
    
    if 'BBB' not in df.columns:
        logging.error("BBB column missing.")
        sys.exit("Critical error: BBB column missing.")

    df = df.dropna(subset=['BBB'])
    for col in ['BRIC_SMILES', 'RINGS_SMILES', 'SIDE_CHAINS_SMILES']:
        if col in df.columns:
            df[col] = df[col].fillna('')
    return df


def clean_labels(df):
    """
    Standardizes the 'BBB' classification labels to ensure consistency.

    - Converts 'BBB' values to uppercase and removes extra spaces.
    - Replaces variations of 'BBB+' and 'BBB-' with standardized labels.
    - Ensures only 'BBB+' and 'BBB-' remain as valid classifications.

    Parameters:
        df (pd.DataFrame): Input DataFrame containing the 'BBB' classification column.

    Returns:
        pd.DataFrame: DataFrame with standardized 'BBB' labels.
    """
    
    df['BBB'] = df['BBB'].astype(str).str.strip().str.upper()
    df['BBB'] = df['BBB'].replace({
        'BBB +': 'BBB+',
        'BBB -': 'BBB-',
        'BBB+': 'BBB+',
        'BBB-': 'BBB-'
    })
    return df


def encode_labels(df):
    """
    Encodes the 'BBB' classification into numerical labels.

    - Maps 'BBB+' to 1 and 'BBB-' to 0 for binary classification.
    - Uses 'LabelEncoder' from scikit-learn to perform encoding.
    - Logs successful encoding or terminates on failure.

    Parameters:
        df (pd.DataFrame): Input DataFrame containing the 'BBB' classification column.

    Returns:
        tuple:
            pd.DataFrame: DataFrame with an additional 'BBB_Label' column.
            LabelEncoder: Fitted LabelEncoder instance for decoding if needed.

    Raises:
        SystemExit: If an error occurs during encoding, logs and exits the program.
    """
    
    try:
        label_mapping = {'BBB+': 1, 'BBB-': 0}
        df['BBB_Label'] = df['BBB'].map(label_mapping)

        # Check if all labels are properly mapped
        if df['BBB_Label'].isnull().any():
            missing_labels = df['BBB'][df['BBB_Label'].isnull()].unique()
            logging.error(f"Unrecognized labels found: {missing_labels}")
            sys.exit(f"Critical error: Unrecognized BBB labels: {missing_labels}")

        logging.info("Labels encoded successfully.")
        return df, label_mapping  # Return mapping dictionary for decoding if needed

    except Exception as e:
        logging.error(f"Error encoding labels: {e}")
        sys.exit(f"Critical error: {e}")


def vectorize_fragments_train_only(df_train):
    """
    Trains CountVectorizers for different molecular fragment types (BRICS, RINGS, and SIDE_CHAINS)
    using training data only.

    For each fragment type, this function:
      - Tokenizes the SMILES strings based on spaces (without converting to lowercase).
      - Fits a CountVectorizer to the tokens.
      - Computes a frequency series that counts the number of rows in which each token appears at least once.
    
    Parameters:
        df_train (pd.DataFrame): Training DataFrame

    Returns:
        dict: A dictionary with keys:
            'BRICS': Tuple containing the CountVectorizer and token frequency Series for BRICS fragments.
            RINGS': Tuple containing the CountVectorizer and token frequency Series for ring fragments.
            SIDECHAINS': Tuple containing the CountVectorizer and token frequency Series for side-chain fragments.
    """
    
    # Helper function to build a CountVectorizer and calculate token frequencies for a given series of strings.
    def build_vectorizer_and_freq(series_of_strings):
        vect = CountVectorizer(token_pattern=r'[^ ]+', lowercase=False)
        mat = vect.fit_transform(series_of_strings.tolist())
        feature_names = vect.get_feature_names_out()
        temp_df = pd.DataFrame(mat.toarray(), columns=feature_names)
        freq_series = (temp_df > 0).sum(axis=0)
        return vect, freq_series

    # Train the vectorizer and compute token frequencies
    brics_vect, brics_freq = build_vectorizer_and_freq(df_train['BRIC_SMILES'])
    rings_vect, rings_freq = build_vectorizer_and_freq(df_train['RINGS_SMILES'])
    side_vect, side_freq = build_vectorizer_and_freq(df_train['SIDE_CHAINS_SMILES'])

    # Return a dictionary containing the trained vectorizers and corresponding token frequency Series.
    return {
        'BRICS': (brics_vect, brics_freq),
        'RINGS': (rings_vect, rings_freq),
        'SIDECHAINS': (side_vect, side_freq)
    }


def apply_fragments(series_of_strings, vectorizer, prefix):
    """
    Applies a pre-fitted CountVectorizer to a Series, returning a DataFrame with prefixed column names.

    For the given series:
      - Transforms the text using the pre-fitted CountVectorizer.
      - Converts the resulting sparse matrix into a DataFrame while retaining the original index.
      - Prepends a specified prefix to each column name for clarity.

    Parameters:
        series_of_strings (pd.Series): Series of space-separated tokens.
        vectorizer (CountVectorizer): Pre-fitted CountVectorizer.
        prefix (str): String prefix to add to each column name.

    Returns:
        pd.DataFrame: DataFrame with token counts and prefixed column names.
    """
    
    mat = vectorizer.transform(series_of_strings.tolist())
    feature_names = vectorizer.get_feature_names_out()
    df_frag = pd.DataFrame(mat.toarray(), columns=feature_names, index=series_of_strings.index)
    df_frag.columns = [f"{prefix}{c}" for c in df_frag.columns]
    return df_frag


def impute_missing_values(X_train: pd.DataFrame, X_val: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Performs KNN-based imputation on missing numeric values in training and validation datasets.

    - Fits the imputer only on 'X_train' to prevent data leakage.
    - Uses 'KNNImputer' to estimate missing values for numeric columns.
    - Ensures no NaN values remain after imputation.
    - Logs imputation details for debugging and traceability.

    Parameters:
        X_train (pd.DataFrame): Training feature matrix.
        X_val (pd.DataFrame): Validation feature matrix.

    Returns:
        tuple:
            - pd.DataFrame: Imputed training data.
            - pd.DataFrame: Imputed validation data.

    Raises:
        SystemExit: If NaNs persist after imputation or an unexpected error occurs.
    """
    
    try:
        # Align columns: Ensure same features in both datasets (fill missing ones with 0)
        X_train, X_val = X_train.align(X_val, join='outer', axis=1, fill_value=0)

        # Select numeric columns for imputation
        numeric_cols = list(X_train.select_dtypes(include=[np.number]).columns)

        if len(numeric_cols) > 1:
            logging.info(f"KNN imputation on {len(numeric_cols)} numeric columns in training set.")

            # Fit imputer ONLY on X_train
            knn_imputer = KNNImputer(n_neighbors=5)
            X_train[numeric_cols] = knn_imputer.fit_transform(X_train[numeric_cols])

            # Apply imputation to X_val without fitting
            X_val[numeric_cols] = knn_imputer.transform(X_val[numeric_cols])

        # Ensure no NaNs remain
        X_train.fillna(0, inplace=True)
        X_val.fillna(0, inplace=True)

        # Final validation check
        if X_train.isnull().sum().sum() > 0 or X_val.isnull().sum().sum() > 0:
            logging.error("NaNs remain after final KNN imputation.")
            sys.exit("Critical error: NaNs remain after imputation.")

        logging.info("All missing values imputed successfully.")
        return X_train, X_val

    except Exception as e:
        logging.error(f"Error during KNN imputation: {e}")
        sys.exit(f"Critical error: {e}")


def train_model(
    X_train,
    y_train,
    X_val=None,
    y_val=None,
    random_seed=DEFAULT_RANDOM_SEED,
    scale_pos_weight=1.0,
    tree_method='gpu_hist',
    predictor='gpu_predictor',
    num_boost_round=1000,
    early_stopping_rounds=None,
    **kwargs
):
    """
    Trains an XGBoost model with specified hyperparameters.
    
    - Prepares training data into an XGBoost DMatrix.
    - Applies validation data if provided.
    - Configures and executes the training process.
    - Implements early stopping if validation data is present.
    - Logs the training process and final completion.
    
    Parameters:
        X_train (pd.DataFrame): Training feature matrix.
        y_train (pd.Series): Training target labels.
        X_val (pd.DataFrame, optional): Validation feature matrix.
        y_val (pd.Series, optional): Validation target labels.
        random_seed (int): Random seed for reproducibility.
        scale_pos_weight (float): Scaling factor for handling class imbalance.
        tree_method (str): XGBoost tree algorithm ('gpu_hist' or 'hist').
        predictor (str): XGBoost predictor method ('gpu_predictor' or 'cpu_predictor').
        num_boost_round (int): Number of boosting rounds.
        early_stopping_rounds (int, optional): Stopping criteria based on validation performance.
        kwargs (dict): Additional XGBoost hyperparameters.
    
    Returns:
        tuple:
            - xgb.Booster: Trained XGBoost model.
            - dict: Evaluation results recorded during training.
    """
    
    logging.info("Starting XGBoost training.")
    
    # Ensure column names are valid for XGBoost
    X_train = sanitize_column_names(X_train)
    
    # Convert training data to DMatrix (XGBoost's optimized format)
    dtrain = xgb.DMatrix(X_train, label=y_train)
    # Monitor training progress
    watchlist = [(dtrain, 'train')]

    # Prepare validation data if provided
    if X_val is not None and y_val is not None and len(y_val) > 0:
        X_val = sanitize_column_names(X_val)
        valid_mask = ~y_val.isnull()
        X_val = X_val[valid_mask]
        y_val = y_val[valid_mask]
        
        # Ensure validation set is not empty after NaN removal
        if y_val.empty:
            logging.error("All validation labels are NaN. Exiting.")
            sys.exit("Critical error: Empty validation set.")
            
        dval = xgb.DMatrix(X_val, label=y_val)
        watchlist.append((dval, 'eval'))
    
    # Define model hyperparameters
    params = {
        'objective': 'binary:logistic',
        'eval_metric': 'auc',
        'seed': random_seed,
        'max_depth': 10,
        'learning_rate': 0.016303454,
        'subsample': 0.682094104,
        'colsample_bytree': 0.667401171,
        'min_child_weight': 1,
        'gamma': 0.257201234,
        'alpha': 0.010582666,
        'lambda': 1.204193852,
        'tree_method': tree_method,
        'predictor': predictor,
        'nthread': -1,
        'scale_pos_weight': scale_pos_weight
    }
    
    # Override default hyperparameters with user-defined values
    for k, v in kwargs.items():
        params[k] = v

    evals_result = {}
    
    # Train the model with provided data and parameters
    booster = xgb.train(
        params=params,
        dtrain=dtrain,
        num_boost_round=num_boost_round,
        evals=watchlist,
        early_stopping_rounds=early_stopping_rounds if (X_val is not None and y_val is not None) else None,
        evals_result=evals_result,
        verbose_eval=False # Suppresses detailed output during training
    )
    logging.info("XGBoost training complete.")
    return booster, evals_result


def evaluate_model(
    booster, 
    X_eval, 
    y_eval, 
    threshold=0.5
):
    """
    Evaluate an XGBoost model's performance on a given dataset using a defined threshold.
    
    - Threshold is constrained between [0.35, 0.65] to prevent extreme decision boundaries.
    - Computes key classification metrics including AUC, MCC, F1 Score, Sensitivity, and Specificity.
    - Generates a confusion matrix and extracts TP, TN, FP, FN.
    
    Parameters:
        booster (xgb.Booster): Trained XGBoost model.
        X_eval (pd.DataFrame): Feature matrix for evaluation.
        y_eval (pd.Series): True labels corresponding to X_eval.
        threshold (float): Decision threshold for classification (default: 0.5).
    
    Returns:
        tuple:
            - dict: Contains computed evaluation metrics.
            - np.array: Predicted probabilities for the positive class.
            - np.array: Binary predictions based on the given threshold.
    """
    
    logging.info("Evaluating model.")
    
    # Ensure threshold is within the defined safe range
    threshold = min(max(threshold, 0.35), 0.65)

    # Sanitize column names to avoid XGBoost compatibility issues
    X_eval = sanitize_column_names(X_eval)
    
    # Remove instances with missing labels
    valid_mask = ~y_eval.isnull()
    X_eval = X_eval[valid_mask]
    y_eval = y_eval[valid_mask]
    
    # Handle case where all labels are NaN after filtering
    if y_eval.empty:
        logging.error("No validation labels remaining.")
        sys.exit("Critical error: Empty eval set.")

    # Convert evaluation data into XGBoost's DMatrix format
    d_eval = xgb.DMatrix(X_eval)
    
    # Predict class probabilities
    y_probs = booster.predict(d_eval)
    
    # Generate binary predictions based on the threshold
    y_pred = (y_probs >= threshold).astype(int)

    # Convert ground truth labels to NumPy array for processing
    y_eval_np = y_eval.values.astype(int)

    # Compute primary classification metrics
    if len(np.unique(y_eval_np)) > 1:
        auc_val = roc_auc_score(y_eval_np, y_probs)
        mcc = matthews_corrcoef(y_eval_np, y_pred)
        f1_ = f1_score(y_eval_np, y_pred)
    else:
        auc_val = np.nan
        mcc = np.nan
        f1_ = np.nan

    # Compute confusion matrix
    cm = confusion_matrix(y_eval_np, y_pred, labels=[0, 1])
    
    # Extract confusion matrix components
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
    else:
        tn = fp = fn = tp = 0
        if len(np.unique(y_eval_np)) == 1:
            if y_eval_np[0] == 0:
                tn = cm[0, 0]
            else:
                tp = cm[0, 0]

    # Compute sensitivity and specificity (handle division by zero cases)
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else np.nan
    specificity = tn / (tn + fp) if (tn + fp) > 0 else np.nan

    # Store evaluation metrics in a dictionary
    metrics = {
        'AUC': auc_val,
        'MCC': mcc,
        'F1 Score': f1_,
        'TP': tp,
        'TN': tn,
        'FP': fp,
        'FN': fn,
        'Sensitivity (SN%)': sensitivity * 100 if not np.isnan(sensitivity) else np.nan,
        'Specificity (SP%)': specificity * 100 if not np.isnan(specificity) else np.nan
    }
    
    logging.info("Model evaluation complete.")
    
    return metrics, y_probs, y_pred


def plot_calibration_curve_func(y_true, y_probs, output_dir):
    """
    Generates and saves a calibration curve for model probability predictions.

    - Compares predicted probabilities with actual fraction of positive cases.
    - Plots a reference diagonal line for perfect calibration.
    - Saves the output figure as 'calibration_curve.png' in the specified directory.

    Parameters:
        y_true (np.array): Array of true binary labels (0 or 1).
        y_probs (np.array): Array of predicted probabilities from the model.
        output_dir (str): Path to the directory where the plot will be saved.

    Returns:
        None: Saves the plot to disk.
    """
    
    fraction_of_positives, mean_predicted_value = calibration_curve(y_true, y_probs, n_bins=10)
    plt.figure()
    plt.plot(mean_predicted_value, fraction_of_positives, "s-", label='Model')
    plt.plot([0, 1], [0, 1], "k--", label='Perfectly calibrated')
    plt.xlabel('Mean predicted probability')
    plt.ylabel('Fraction of positives')
    plt.title('Calibration Curve')
    plt.legend(loc='lower right')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'calibration_curve.png'), dpi=300)
    plt.close()


def plot_roc_curve(y_val, y_probs, output_dir):
    """
    Generates and saves an ROC (Receiver Operating Characteristic) curve.

    - Computes the True Positive Rate (TPR) and False Positive Rate (FPR).
    - Plots a reference diagonal line indicating a random classifier.
    - Saves the output figure as 'roc_curve.png' in the specified directory.

    Parameters:
        y_val (pd.Series or np.array): Array of true binary labels (0 or 1).
        y_probs (np.array): Array of predicted probabilities from the model.
        output_dir (str): Path to the directory where the plot will be saved.

    Returns:
        None: Saves the plot to disk.
    """
    
    try:
        if len(np.unique(y_val)) > 1:
            fpr, tpr, _ = roc_curve(y_val, y_probs)
            auc_val = roc_auc_score(y_val, y_probs)
            plt.figure()
            plt.plot(fpr, tpr, label=f"AUC = {auc_val:.3f}")
            plt.plot([0,1],[0,1],'k--')
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('ROC Curve')
            plt.legend(loc='lower right')
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'roc_curve.png'), dpi=300)
            plt.close()
        else:
            logging.warning("Single-class data: skipping ROC plot.")
    except Exception as e:
        logging.error(f"Error plotting ROC: {e}")
        sys.exit(f"Critical error: {e}")


def save_confusion_matrix(cm, output_path):
    """
    Saves a confusion matrix as a CSV file.

    - Converts the NumPy array into a labeled DataFrame.
    - Saves the DataFrame as a CSV file with headers for easy interpretation.

    Parameters:
        cm (np.array): Confusion matrix as a 2D NumPy array.
        output_path (str): Path where the CSV file will be saved.

    Returns:
        None: Saves the confusion matrix to disk.
    """
    
    cm_df = pd.DataFrame(cm,
                         index=['Actual BBB-', 'Actual BBB+'],
                         columns=['Predicted BBB-', 'Predicted BBB+'])
    cm_df.to_csv(output_path, index=True)
    logging.info(f"Confusion matrix saved at {output_path}")


def save_model(booster, output_dir, model_name='xgboost_model.json'):
    """
    Saves an XGBoost booster model to disk.

    - Converts the trained booster into a JSON file format.
    - Ensures the specified output directory exists before saving.
    - Logs the save operation and provides a console message on success.
    - Handles potential errors and exits if saving fails.

    Parameters:
        booster (xgb.Booster): The trained XGBoost model to be saved.
        output_dir (str): Path to the directory where the model will be stored.
        model_name (str): Name of the output model file (default: 'xgboost_model.json').

    Returns:
        None: Saves the model to disk.
    """
    
    try:
        model_path = os.path.join(output_dir, model_name)
        booster.save_model(model_path)
        logging.info(f"Model saved at {model_path}")
        print(f"Model saved successfully at: {model_path}")
    except Exception as e:
        logging.error(f"Error saving model: {e}")
        sys.exit(f"Critical error: {e}")


def perform_shap_analysis(booster, X_train, X_val, y_val, output_dir):
    """
    Performs SHAP (SHapley Additive exPlanations) analysis to interpret model predictions.

    - Computes SHAP values to assess feature influence on BBB permeability.
    - Generates SHAP summary plots for all features and key subsets (BRICS, RINGS, SIDECHAINS).
    - Saves SHAP values as CSV files for further inspection.
    - Produces XGBoost feature importance plots for the top 50 features.
    - Creates a correlation heatmap for the most important features.
    - Handles missing values and ensures proper formatting of feature names.

    Parameters:
        booster (xgb.Booster): Trained XGBoost model.
        X_train (pd.DataFrame): Training dataset (for feature correlation).
        X_val (pd.DataFrame): Validation dataset (for SHAP analysis).
        y_val (pd.Series): Validation target labels.
        output_dir (str): Path to save analysis outputs.

    Returns:
        None: Saves multiple plots and data files to the specified directory.
    """
    
    try:
        logging.info("Starting SHAP analysis.")
        
        # Create output directory for SHAP analysis
        analysis_dir = os.path.join(output_dir, 'analysis')
        os.makedirs(analysis_dir, exist_ok=True)

        # Select only numeric columns from X_val for SHAP computation
        X_val_num = X_val.select_dtypes(include=[np.number]).copy()
        if X_val_num.empty:
            logging.warning("No numeric columns for SHAP. Skipping SHAP analysis.")
            return

        # Compute SHAP values using TreeExplainer
        explainer = shap.TreeExplainer(booster, feature_perturbation='interventional')
        arr_val = X_val_num.values.astype(np.float32)
        shap_values_pos = explainer.shap_values(arr_val, check_additivity=False)

        # Handle binary classification case (SHAP returns list of two arrays)
        if isinstance(shap_values_pos, list) and len(shap_values_pos) == 2:
            shap_values_pos = shap_values_pos[1]

        # Flip sign so + => 'drive BBB+'
        shap_values_pos = -1 * shap_values_pos
        shap_df = pd.DataFrame(shap_values_pos, columns=X_val_num.columns)
        shap_df.to_csv(os.path.join(analysis_dir, 'shap_values.csv'), index=False)

        # Generate SHAP summary plot
        if shap_df.shape[1] > 0:
            plt.figure(figsize=(12, 10))
            shap.summary_plot(shap_df.values, X_val_num, show=False)
            
            # Format y-axis labels
            ax = plt.gca()
            yticklabels = [label.get_text().replace('_', ' ') for label in ax.get_yticklabels()]
            ax.set_yticklabels(yticklabels)

            plt.title('SHAP Summary Plot', fontsize=16)
            plt.tight_layout()
            plt.savefig(os.path.join(analysis_dir, 'shap_summary_plot.png'), dpi=300)
            plt.close()

        # Extract feature importances from the trained model
        imp_dict = booster.get_score(importance_type='weight')
        mapped_importances = {}
        train_cols = list(X_train.columns)
        
        # Map feature indices to column names
        for k, v in imp_dict.items():
            if k.startswith('f'):
                try:
                    idx = int(k[1:])
                    if idx < len(train_cols):
                        col_name = train_cols[idx]
                        mapped_importances[col_name] = v
                except:
                    logging.warning(f"Cannot parse feature index: {k}")
            else:
                if k in train_cols:
                    mapped_importances[k] = v

        # Convert feature importance dictionary to pandas Series
        fi_series = pd.Series(mapped_importances).fillna(0).sort_values(ascending=False)
        fi_top50 = fi_series.head(50)
        
        # Plot feature importance for the top 50 features
        if len(fi_top50) > 0:
            plt.figure(figsize=(12,8))
            sns.barplot(x=fi_top50.values, y=[revert_brics_naming(i) for i in fi_top50.index], palette='viridis')
            
            ax = plt.gca()
            yticklabels = [label.get_text().replace('_', ' ') for label in ax.get_yticklabels()]
            ax.set_yticklabels(yticklabels)
            
            plt.title('Top 50 Feature Importances (All Features)')
            plt.xlabel('Importance Score')
            plt.ylabel('Features')
            plt.tight_layout()
            plt.savefig(os.path.join(analysis_dir, 'feature_importance_plot_ALL.png'), dpi=300)
            plt.close()
        
        # Define feature subsets for focused analysis
        all_cols = X_train.columns
        brics_cols = [c for c in all_cols if c.startswith('BRICS_')]
        rings_cols = [c for c in all_cols if c.startswith('RINGS_')]
        side_cols = [c for c in all_cols if c.startswith('SIDECHAINS_')]
        all_except_cols = [c for c in all_cols if c not in set(brics_cols + rings_cols + side_cols)]

        # Function to plot feature importance for specific subsets
        def plot_feature_importance_subset(subset_cols, name_suffix):
            if not subset_cols:
                logging.info(f"No columns for subset: {name_suffix}")
                return
            
            subset_fi = fi_series[fi_series.index.isin(subset_cols)]
            
            if subset_fi.empty:
                logging.warning(f"No feature importance data for subset: {name_suffix}")
                return
            
            subset_fi_top50 = subset_fi.head(50)

            plt.figure(figsize=(12, 8))
            sns.barplot(x=subset_fi_top50.values, y=[revert_brics_naming(i) for i in subset_fi_top50.index], palette='viridis')

            ax = plt.gca()
            yticklabels = [label.get_text().replace('_', ' ') for label in ax.get_yticklabels()]
            ax.set_yticklabels(yticklabels)
            title_name_suffix = name_suffix.replace('_', ' ')
            
            plt.title(f'Top 50 Feature Importances ({title_name_suffix})', fontsize=16)
            plt.xlabel('Importance Score')
            plt.ylabel('Features')
            
            plt.tight_layout()
            plt.savefig(os.path.join(analysis_dir, f'feature_importance_plot_{name_suffix}.png'), dpi=300)
            plt.close()
        
        # Generate Feature Importance Plots for Subsets
        plot_feature_importance_subset(all_except_cols, "not_FRAGMENTS")
        plot_feature_importance_subset(brics_cols, "only_BRICS")
        plot_feature_importance_subset(rings_cols, "only_RINGS")
        plot_feature_importance_subset(side_cols, "only_SIDE_CHAINS")
        
        # Generate correlation heatmap for top 50 most important features
        top_corr_features = fi_series.head(50).index.tolist()
        top_corr_features = [f for f in top_corr_features if f in X_train.columns]
        
        if len(top_corr_features) > 1:
            corr_mat = X_train[top_corr_features].corr()
            mask = np.triu(np.ones_like(corr_mat, dtype=bool))
            sns.set(font_scale=0.5)
            plt.figure(figsize=(16, 14))
            
            formatted_columns = [i.replace('_', ' ') for i in corr_mat.columns]
            formatted_index = [i.replace('_', ' ') for i in corr_mat.index]
            
            sns.heatmap(
                corr_mat, annot=True, fmt=".2f", cmap='coolwarm',
                square=True, mask=mask, linewidths=0.5, linecolor='gray',
                xticklabels=formatted_columns, yticklabels=formatted_index
            )
            
            plt.title('Correlation Heatmap (Top 50 Features)')
            plt.tight_layout()
            plt.savefig(os.path.join(analysis_dir, 'correlation_heatmap.png'), dpi=300)
            plt.close()

        # Additional SHAP subset analyses
        all_cols = X_val_num.columns
        brics_cols = [c for c in all_cols if c.startswith('BRICS_')]
        rings_cols = [c for c in all_cols if c.startswith('RINGS_')]
        side_cols = [c for c in all_cols if c.startswith('SIDECHAINS_')]
        all_except_cols = [c for c in all_cols if c not in set(brics_cols + rings_cols + side_cols)]

        # Function to generate SHAP summary plots for feature subsets
        def shap_subset_plot(subset_cols, name_suffix):
            if not subset_cols:
                logging.info(f"No columns for subset: {name_suffix}")
                return
            
            sub_shap_df = shap_df[subset_cols].copy()
            sub_X_val = X_val_num[subset_cols].copy()
            
            if sub_shap_df.empty:
                return
            
            sub_shap_df.to_csv(os.path.join(analysis_dir, f'shap_values_{name_suffix}.csv'), index=False)

            plt.figure(figsize=(12, 10))
            shap.summary_plot(sub_shap_df.values, sub_X_val, show=False)
            
            ax = plt.gca()
            yticklabels = [label.get_text().replace('_', ' ') for label in ax.get_yticklabels()]
            ax.set_yticklabels(yticklabels)
            title_name_suffix = name_suffix.replace('_', ' ')

            plt.title(f'SHAP Plot ({title_name_suffix})', fontsize=16)
            plt.tight_layout()
            plt.savefig(os.path.join(analysis_dir, f'shap_summary_plot_{name_suffix}.png'), dpi=300)
            plt.close()

        # Generate SHAP subset analyses
        shap_subset_plot(all_except_cols, "not_FRAGMENTS")
        shap_subset_plot(brics_cols, "only_BRICS")
        shap_subset_plot(rings_cols, "only_RINGS")
        shap_subset_plot(side_cols, "only_SIDE_CHAINS")

        logging.info("SHAP analysis complete.")
    
    except Exception as e:
        logging.error(f"Error in SHAP analysis: {e}")
        sys.exit(f"Critical error: {e}")


def run_optuna_optimization(
    df_train,
    y_train,
    df_val,
    dict_of_vectorizers,
    optimize_metric,
    n_trials,
    tree_method,
    predictor,
    base_output_dir,
    n_folds=DEFAULT_N_FOLDS,
    balance_method=None,
    random_seed_main=DEFAULT_RANDOM_SEED
):
    """
    Runs Optuna optimization for an XGBoost pipeline with cross-validation.
    
    This function performs Bayesian optimization to tune hyperparameters for an XGBoost model,
    while also optimizing fragment vectorization frequency thresholds. It uses cross-validation
    to evaluate the performance (based on a chosen metric) and then re-trains a final model with the best
    hyperparameters on the full training set and a hold-out validation set.
    
    Parameters:
        df_train (pd.DataFrame): Training DataFrame containing molecular data and fragment SMILES.
        y_train (pd.Series): Target labels for the training data.
        df_val (pd.DataFrame): Validation DataFrame, must contain non-null 'BBB_Label' values.
        dict_of_vectorizers (dict): Dictionary with keys 'BRICS', 'RINGS', and 'SIDECHAINS' containing
            tuples of (CountVectorizer, frequency Series) for each fragment type.
        optimize_metric (str): The metric to optimize (e.g., "f1", "auc", etc.).
        n_trials (int): Number of Optuna trials to perform.
        tree_method (str): XGBoost tree method (e.g., 'gpu_hist' or 'hist').
        predictor (str): XGBoost predictor type (e.g., 'gpu_predictor' or 'cpu_predictor').
        base_output_dir (str): Base directory where output files and logs will be saved.
        n_folds (int, optional): Number of folds for cross-validation (default is DEFAULT_N_FOLDS).
        balance_method (str, optional): Data balancing method to apply (e.g., 'smote', 'smoteenn', or 'smotetomek').
        random_seed_main (int, optional): Random seed for reproducibility (default is DEFAULT_RANDOM_SEED).
    
    Returns:
        tuple: (study, final_metrics)
            study (optuna.Study): The Optuna study object containing all trial results.
            final_metrics (dict): Final evaluation metrics from the best model on the hold-out validation set.
    """
    
    # Drop rows from validation set where the target 'BBB_Label' is missing.
    df_val = df_val.dropna(subset=['BBB_Label'])
    if df_val.empty:
        sys.exit("Critical error: Validation set is empty after dropping NaNs in BBB_Label.")

    # Extract target labels from validation set.
    y_val = df_val['BBB_Label']
    # Create subfolder for the current optimization metric.
    metric_subfolder = os.path.join(base_output_dir, optimize_metric.replace(" ", "_").lower())
    os.makedirs(metric_subfolder, exist_ok=True)

    # Set up logging for the current metric subfolder.
    log_file = os.path.join(metric_subfolder, 'model_training.log')
    setup_logging(log_file)

    # Retrieve vectorizers and frequency Series for each fragment type.
    brics_vect, brics_freq = dict_of_vectorizers['BRICS']
    rings_vect, rings_freq = dict_of_vectorizers['RINGS']
    side_vect, side_freq = dict_of_vectorizers['SIDECHAINS']

    # Create an Optuna study to maximize the chosen metric.
    study = optuna.create_study(direction='maximize')

    def objective(trial):
        # Use the main random seed for the trial.
        random_seed_trial = random_seed_main

        # Suggest fixed minimum frequency values for each fragment type.
        brics_min_freq = trial.suggest_int("brics_min_freq", 3, 3)      # Temp 5: ideal (1,6)
        rings_min_freq = trial.suggest_int("rings_min_freq", 3, 3)      # Temp 5: ideal (1,6)
        side_min_freq = trial.suggest_int("side_min_freq", 5, 5)        # Temp 5: ideal (2,10)

        # Determine the list of tokens to keep based on the frequency thresholds.
        brics_keep_list = brics_freq[brics_freq >= brics_min_freq].index.tolist()
        rings_keep_list = rings_freq[rings_freq >= rings_min_freq].index.tolist()
        side_keep_list = side_freq[side_freq >= side_min_freq].index.tolist()

        # Apply the pre-fitted vectorizers to the SMILES strings from the training data.
        brics_df_train = apply_fragments(df_train['BRIC_SMILES'], brics_vect, 'BRICS_')
        # Filter columns to retain only tokens that meet the frequency threshold.
        brics_df_train = brics_df_train[[c for c in brics_df_train.columns if c.replace("BRICS_", "") in brics_keep_list]]
        rings_df_train = apply_fragments(df_train['RINGS_SMILES'], rings_vect, 'RINGS_')
        rings_df_train = rings_df_train[[c for c in rings_df_train.columns if c.replace("RINGS_", "") in rings_keep_list]]
        side_df_train = apply_fragments(df_train['SIDE_CHAINS_SMILES'], side_vect, 'SIDECHAINS_')
        side_df_train = side_df_train[[c for c in side_df_train.columns if c.replace("SIDECHAINS_", "") in side_keep_list]]

        # Prepare the numeric features from the training data by dropping non-numeric columns.
        numeric_train = df_train.drop(columns=[
            'NO.', 'BBB', 'LogBB', 'Group', 'BBB_Label',
            'BRIC_SMILES', 'RINGS_SMILES', 'SIDE_CHAINS_SMILES', 'SMILES'
        ], errors='ignore')
        # Concatenate numeric features with the fragment features.
        X_train_total = pd.concat([numeric_train, brics_df_train, rings_df_train, side_df_train], axis=1)
        # Remove duplicate columns and convert all data to numeric.
        X_train_total = X_train_total.loc[:, ~X_train_total.columns.duplicated()].copy()
        X_train_total = X_train_total.apply(pd.to_numeric, errors='coerce')

        # Suggest hyperparameters for XGBoost.
        num_boost_round = trial.suggest_int('num_boost_round', 100, 2000)  # Number of boosting rounds
        early_stopping_r = trial.suggest_int('early_stopping_rounds', 10, 300)  # Early stopping rounds
        max_depth = trial.suggest_int('max_depth', 3, 30)  # Maximum tree depth
        learning_rate = trial.suggest_float('learning_rate', 0.001, 0.3, log=True)  # Learning rate
        subsample = trial.suggest_float('subsample', 0.4, 1.0)  # Subsample ratio of the training instance
        colsample_bytree = trial.suggest_float('colsample_bytree', 0.4, 1.0)  # Subsample ratio of columns when constructing each tree
        min_child_weight = trial.suggest_int('min_child_weight', 1, 15)  # Minimum sum of instance weight needed in a child
        gamma = trial.suggest_float('gamma', 0.0, 5.0)  # Minimum loss reduction required to make a further partition
        reg_alpha = trial.suggest_float('reg_alpha', 0.0, 5.0)  # L1 regularization term
        reg_lambda = trial.suggest_float('reg_lambda', 0.0, 5.0)  # L2 regularization term

        # Set up Stratified K-Fold cross-validation.
        skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_seed_trial)
        cv_recalls = []
        # Loop through each fold for cross-validation.
        for fold_train_idx, fold_test_idx in skf.split(X_train_total, y_train):
            # Split the data into training and testing folds.
            X_tr_fold = X_train_total.iloc[fold_train_idx].copy()
            y_tr_fold = y_train.iloc[fold_train_idx]
            X_te_fold = X_train_total.iloc[fold_test_idx].copy()
            y_te_fold = y_train.iloc[fold_test_idx]

            # Impute missing values for the current fold.
            X_tr_fold_imp, X_te_fold_imp = impute_missing_values(X_tr_fold, X_te_fold)
            # Sanitize column names for XGBoost compatibility.
            X_tr_fold_imp = sanitize_column_names(X_tr_fold_imp)
            X_te_fold_imp = sanitize_column_names(X_te_fold_imp)

            # Apply data balancing if specified.
            if balance_method is not None:
                if balance_method.lower() == 'smote':
                    sampler = SMOTE(random_state=random_seed_trial)
                elif balance_method.lower() == 'smoteenn':
                    sampler = SMOTEENN(random_state=random_seed_trial)
                elif balance_method.lower() == 'smotetomek':
                    sampler = SMOTETomek(random_state=random_seed_trial)
                else:
                    sampler = None
            else:
                sampler = None

            if sampler is not None:
                X_tr_bal, y_tr_bal = sampler.fit_resample(X_tr_fold_imp, y_tr_fold)
            else:
                X_tr_bal, y_tr_bal = X_tr_fold_imp, y_tr_fold

            # Calculate scale_pos_weight for balancing classes.
            pos_count_fold = (y_tr_bal == 1).sum()
            neg_count_fold = (y_tr_bal == 0).sum()
            scale_pos_weight_fold = neg_count_fold / pos_count_fold if pos_count_fold > 0 else 1.0

            # Train an XGBoost model on the balanced fold data using the suggested hyperparameters.
            booster_cv, _ = train_model(
                X_tr_bal, y_tr_bal,
                X_te_fold_imp, y_te_fold,
                random_seed=random_seed_trial,
                tree_method=tree_method,
                predictor=predictor,
                scale_pos_weight=scale_pos_weight_fold,
                num_boost_round=num_boost_round,
                early_stopping_rounds=early_stopping_r,
                max_depth=max_depth,
                learning_rate=learning_rate,
                subsample=subsample,
                colsample_bytree=colsample_bytree,
                min_child_weight=min_child_weight,
                gamma=gamma,
                alpha=reg_alpha,
                lambda_=reg_lambda
            )
            # Predict on the test fold.
            d_fold_test = xgb.DMatrix(X_te_fold_imp)
            y_probs_cv = booster_cv.predict(d_fold_test)
            y_pred_cv = (y_probs_cv >= 0.5).astype(int)
            # Compute recall for the current fold and store it.
            recall_cv = recall_score(y_te_fold, y_pred_cv, zero_division=0)
            cv_recalls.append(recall_cv)

        # Calculate average cross-validation recall.
        avg_cv_recall = float(np.mean(cv_recalls))

        # Evaluate final performance on the hold-out validation set.
        brics_df_val = apply_fragments(df_val['BRIC_SMILES'], brics_vect, 'BRICS_')
        brics_df_val = brics_df_val[[c for c in brics_df_val.columns if c.replace("BRICS_", "") in brics_keep_list]]
        rings_df_val = apply_fragments(df_val['RINGS_SMILES'], rings_vect, 'RINGS_')
        rings_df_val = rings_df_val[[c for c in rings_df_val.columns if c.replace("RINGS_", "") in rings_keep_list]]
        side_df_val = apply_fragments(df_val['SIDE_CHAINS_SMILES'], side_vect, 'SIDECHAINS_')
        side_df_val = side_df_val[[c for c in side_df_val.columns if c.replace("SIDECHAINS_", "") in side_keep_list]]

        # Prepare numeric features from the validation set.
        numeric_val = df_val.drop(columns=[
            'NO.', 'BBB', 'LogBB', 'Group', 'BBB_Label',
            'BRIC_SMILES', 'RINGS_SMILES', 'SIDE_CHAINS_SMILES', 'SMILES'
        ], errors='ignore')
        # Concatenate numeric and fragment features for the full validation set.
        X_val_full = pd.concat([numeric_val, brics_df_val, rings_df_val, side_df_val], axis=1)
        X_val_full = X_val_full.loc[:, ~X_val_full.columns.duplicated()].copy()
        X_val_full = X_val_full.apply(pd.to_numeric, errors='coerce')

        # Impute missing values in both training and validation sets.
        X_train_total_imp, X_val_imp = impute_missing_values(X_train_total, X_val_full)
        # Sanitize column names.
        X_train_total_imp = sanitize_column_names(X_train_total_imp)
        X_val_imp = sanitize_column_names(X_val_imp)

        # Apply balancing to the full training set if specified.
        if balance_method is not None:
            if balance_method.lower() == 'smote':
                sampler = SMOTE(random_state=random_seed_trial)
            elif balance_method.lower() == 'smoteenn':
                sampler = SMOTEENN(random_state=random_seed_trial)
            elif balance_method.lower() == 'smotetomek':
                sampler = SMOTETomek(random_state=random_seed_trial)
            else:
                sampler = None
        else:
            sampler = None

        if sampler is not None:
            # Summarize class counts before balancing.
            pos_before = (y_train == 1).sum()
            neg_before = (y_train == 0).sum()

            X_train_final, y_train_final = sampler.fit_resample(X_train_total_imp, y_train)

            pos_after = (y_train_final == 1).sum()
            neg_after = (y_train_final == 0).sum()
        else:
            X_train_final, y_train_final = X_train_total_imp, y_train

        # Calculate final scale_pos_weight.
        pos_count_final = (y_train_final == 1).sum()
        neg_count_final = (y_train_final == 0).sum()
        spw_final = neg_count_final / pos_count_final if pos_count_final > 0 else 1.0

        # Train the final model on the full training set with the current trial's hyperparameters.
        booster_final, _ = train_model(
            X_train_final,
            y_train_final,
            X_val_imp,
            y_val,
            random_seed=random_seed_trial,
            tree_method=tree_method,
            predictor=predictor,
            scale_pos_weight=spw_final,
            num_boost_round=num_boost_round,
            early_stopping_rounds=early_stopping_r,
            max_depth=max_depth,
            learning_rate=learning_rate,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            min_child_weight=min_child_weight,
            gamma=gamma,
            alpha=reg_alpha,
            lambda_=reg_lambda
        )
        # Predict on the full validation set.
        d_val_final = xgb.DMatrix(X_val_imp)
        y_probs_val = booster_final.predict(d_val_final)
        y_pred_val = (y_probs_val >= 0.5).astype(int)

        # Compute the chosen evaluation metric on the validation set.
        if len(np.unique(y_val)) > 1:
            if optimize_metric.lower() == 'f1':
                metric_val = f1_score(y_val, y_pred_val)
            elif optimize_metric.lower() == 'balanced accuracy':
                metric_val = balanced_accuracy_score(y_val, y_pred_val)
            elif optimize_metric.lower() == 'precision':
                metric_val = precision_score(y_val, y_pred_val, zero_division=0)
            elif optimize_metric.lower() == 'recall':
                metric_val = recall_score(y_val, y_pred_val, zero_division=0)
            elif optimize_metric.lower() == 'accuracy':
                metric_val = accuracy_score(y_val, y_pred_val)
            elif optimize_metric.lower() == 'auc':
                metric_val = roc_auc_score(y_val, y_probs_val)
            else:
                metric_val = f1_score(y_val, y_pred_val)
        else:
            metric_val = 0.5

        # Store cross-validation recall and final validation metric in trial user attributes.
        trial.set_user_attr("cv_recall", avg_cv_recall)
        trial.set_user_attr("val_metric", metric_val)
        return metric_val

    # Optimize the objective function for the specified number of trials.
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    # Re-train final model with best hyperparameters
    best_trial = study.best_trial
    best_val_metric = study.best_value
    best_params = best_trial.params

    # Collect user attributes from all completed trials for comparison.
    all_data = []
    for t in study.trials:
        if t.state == optuna.trial.TrialState.COMPLETE and t.value is not None:
            cv_recall_ = t.user_attrs.get("cv_recall", None)
            val_metric_ = t.user_attrs.get("val_metric", None)
            all_data.append({
                "trial_number": t.number,
                "cv_recall": cv_recall_,
                "val_metric": val_metric_,
                "hyperparams": t.params
            })

    df_comparison = pd.DataFrame(all_data)
    
    # Plot a comparison chart: CV recall (x-axis) vs. final validation metric (y-axis).        
    if not df_comparison.empty and "cv_recall" in df_comparison.columns and "val_metric" in df_comparison.columns:
        # Compute the difference between test and validation metrics
        df_comparison["diff_metric"] = df_comparison["cv_recall"] - df_comparison["val_metric"]
        
        plt.figure(figsize=(7, 5))
        
        # Scatter plot: x-axis is the validation metric; y-axis is the difference (Test - Validation)
        plt.scatter(df_comparison["val_metric"], df_comparison["diff_metric"],
                    c='blue', alpha=0.7, label='All Trials')
        
        # Identify and highlight the trial(s) with the highest validation metric
        best_val = df_comparison["val_metric"].max()
        best_rows = df_comparison[df_comparison["val_metric"] == best_val]
        plt.scatter(best_rows["val_metric"], best_rows["diff_metric"],
                    c='red', s=100, label='Best')
        
        plt.xlabel(f"Validation {optimize_metric.capitalize()}")
        plt.ylabel("Test - Validation Difference")
        plt.title("Comparison Chart: Validation Metric vs. Test-Validation Difference")
        plt.legend()
        plt.tight_layout()
        
        chart_path = os.path.join(metric_subfolder, "comparison_evaluation_chart.png")
        plt.savefig(chart_path, dpi=300)
        plt.close()

    # Re-build the final training set using the best trial's frequency thresholds.
    best_brics_min = best_params.get("brics_min_freq", 1)
    best_rings_min = best_params.get("rings_min_freq", 1)
    best_side_min = best_params.get("side_min_freq", 1)

    brics_keep_list = brics_freq[brics_freq >= best_brics_min].index.tolist()
    rings_keep_list = rings_freq[rings_freq >= best_rings_min].index.tolist()
    side_keep_list = side_freq[side_freq >= best_side_min].index.tolist()

    # Save the best frequency thresholds and vectorizers for future use.
    joblib.dump(brics_keep_list, os.path.join(metric_subfolder, "kept_tokens_brics.joblib"))
    joblib.dump(rings_keep_list, os.path.join(metric_subfolder, "kept_tokens_rings.joblib"))
    joblib.dump(side_keep_list, os.path.join(metric_subfolder, "kept_tokens_sidechains.joblib"))
    joblib.dump(brics_vect, os.path.join(metric_subfolder, "vectorizer_brics.joblib"))
    joblib.dump(rings_vect, os.path.join(metric_subfolder, "vectorizer_rings.joblib"))
    joblib.dump(side_vect, os.path.join(metric_subfolder, "vectorizer_sidechains.joblib"))

    # Build the final training data with the selected fragment features.
    brics_df_train = apply_fragments(df_train['BRIC_SMILES'], brics_vect, 'BRICS_')
    brics_df_train = brics_df_train[[c for c in brics_df_train.columns if c.replace("BRICS_", "") in brics_keep_list]]
    rings_df_train = apply_fragments(df_train['RINGS_SMILES'], rings_vect, 'RINGS_')
    rings_df_train = rings_df_train[[c for c in rings_df_train.columns if c.replace("RINGS_", "") in rings_keep_list]]
    side_df_train = apply_fragments(df_train['SIDE_CHAINS_SMILES'], side_vect, 'SIDECHAINS_')
    side_df_train = side_df_train[[c for c in side_df_train.columns if c.replace("SIDECHAINS_", "") in side_keep_list]]

    numeric_train = df_train.drop(columns=[
        'NO.', 'BBB', 'LogBB', 'Group', 'BBB_Label',
        'BRIC_SMILES', 'RINGS_SMILES', 'SIDE_CHAINS_SMILES', 'SMILES'
    ], errors='ignore')
    X_train_total = pd.concat([numeric_train, brics_df_train, rings_df_train, side_df_train], axis=1)
    X_train_total = X_train_total.loc[:, ~X_train_total.columns.duplicated()].copy()
    X_train_total = X_train_total.apply(pd.to_numeric, errors='coerce')
    y_train_series = df_train['BBB_Label']

    # Clean the validation set.
    df_val_clean = df_val.dropna(subset=['BBB_Label'])
    y_val_clean = df_val_clean['BBB_Label']

    brics_df_val = apply_fragments(df_val_clean['BRIC_SMILES'], brics_vect, 'BRICS_')
    brics_df_val = brics_df_val[[c for c in brics_df_val.columns if c.replace("BRICS_", "") in brics_keep_list]]
    rings_df_val = apply_fragments(df_val_clean['RINGS_SMILES'], rings_vect, 'RINGS_')
    rings_df_val = rings_df_val[[c for c in rings_df_val.columns if c.replace("RINGS_", "") in rings_keep_list]]
    side_df_val = apply_fragments(df_val_clean['SIDE_CHAINS_SMILES'], side_vect, 'SIDECHAINS_')
    side_df_val = side_df_val[[c for c in side_df_val.columns if c.replace("SIDECHAINS_", "") in side_keep_list]]

    numeric_val = df_val_clean.drop(columns=[
        'NO.', 'BBB', 'LogBB', 'Group', 'BBB_Label',
        'BRIC_SMILES', 'RINGS_SMILES', 'SIDE_CHAINS_SMILES', 'SMILES'
    ], errors='ignore')
    X_val_total = pd.concat([numeric_val, brics_df_val, rings_df_val, side_df_val], axis=1)
    X_val_total = X_val_total.loc[:, ~X_val_total.columns.duplicated()].copy()
    X_val_total = X_val_total.apply(pd.to_numeric, errors='coerce')

    # Save the training and validation matrices for record keeping.
    train_matrix_csv = os.path.join(metric_subfolder, "train_matrix.csv")
    X_train_total.to_csv(train_matrix_csv, index=True)
    val_matrix_csv = os.path.join(metric_subfolder, "val_matrix.csv")
    X_val_total.to_csv(val_matrix_csv, index=True)

    # Impute missing values and sanitize column names.
    X_train_imp, X_val_imp = impute_missing_values(X_train_total, X_val_total)
    X_train_imp = sanitize_column_names(X_train_imp)
    X_val_imp = sanitize_column_names(X_val_imp)

    # Apply balancing to the final training set if specified.
    if balance_method is not None:
        if balance_method.lower() == 'smote':
            sampler = SMOTE(random_state=DEFAULT_RANDOM_SEED)
        elif balance_method.lower() == 'smoteenn':
            sampler = SMOTEENN(random_state=DEFAULT_RANDOM_SEED)
        elif balance_method.lower() == 'smotetomek':
            sampler = SMOTETomek(random_state=DEFAULT_RANDOM_SEED)
        else:
            sampler = None
    else:
        sampler = None

    pos_before = (y_train_series == 1).sum()
    neg_before = (y_train_series == 0).sum()

    if sampler is not None:
        X_train_final, y_train_final = sampler.fit_resample(X_train_imp, y_train_series)
    else:
        X_train_final, y_train_final = X_train_imp, y_train_series

    pos_count_final = (y_train_final == 1).sum()
    neg_count_final = (y_train_final == 0).sum()
    spw_final = neg_count_final / pos_count_final if pos_count_final > 0 else 1.0

    # Extract final best hyperparameters from the best trial.
    final_num_round = best_params.get('num_boost_round', 1000)
    final_esr = best_params.get('early_stopping_rounds', 50)
    final_max_depth = best_params.get('max_depth', 10)
    final_lr = best_params.get('learning_rate', 0.01)
    final_subsamp = best_params.get('subsample', 0.8)
    final_colsample = best_params.get('colsample_bytree', 0.8)
    final_min_child_weight = best_params.get('min_child_weight', 1)
    final_gamma = best_params.get('gamma', 0.0)
    final_alpha = best_params.get('reg_alpha', 0.0)
    final_lambda = best_params.get('reg_lambda', 1.0)

    # Train the final best model using the final training set and best hyperparameters.
    booster_best, _ = train_model(
        X_train_final,
        y_train_final,
        X_val_imp,
        y_val_clean,
        random_seed=DEFAULT_RANDOM_SEED,
        tree_method=tree_method,
        predictor=predictor,
        scale_pos_weight=spw_final,
        num_boost_round=final_num_round,
        early_stopping_rounds=final_esr,
        max_depth=final_max_depth,
        learning_rate=final_lr,
        subsample=final_subsamp,
        colsample_bytree=final_colsample,
        min_child_weight=final_min_child_weight,
        gamma=final_gamma,
        alpha=final_alpha,
        lambda_=final_lambda
    )

    # Determine the best classification threshold based on balanced accuracy on the training set.
    d_train_b = xgb.DMatrix(X_train_final)
    train_probs = booster_best.predict(d_train_b)
    y_train_arr = y_train_final.values.astype(int)
    best_threshold = 0.5
    best_score = -1
    for th in np.linspace(0.01, 0.99, 99):
        th_clamped = min(max(th, 0.35), 0.65)
        th_pred = (train_probs >= th_clamped).astype(int)
        score = balanced_accuracy_score(y_train_arr, th_pred)
        if score > best_score:
            best_score = score
            best_threshold = th_clamped

    # Evaluate the final model on the validation set using the best threshold.
    final_metrics, y_probs_val, y_pred_val = evaluate_model(booster_best, X_val_imp, y_val_clean, threshold=best_threshold)
    cm = confusion_matrix(y_val_clean, y_pred_val, labels=[0, 1])
    cm_path = os.path.join(metric_subfolder, 'validation_confusion_matrix_best.csv')
    save_confusion_matrix(cm, cm_path)
    pd.DataFrame([final_metrics]).to_csv(os.path.join(metric_subfolder, 'validation_metrics_best.csv'), index=False)

    # Save the best threshold.
    with open(os.path.join(metric_subfolder, 'best_threshold.txt'), 'w') as f:
        f.write(str(best_threshold))

    # Plot ROC and calibration curves, and perform SHAP analysis for feature importance.
    plot_roc_curve(y_val_clean, y_probs_val, metric_subfolder)
    plot_calibration_curve_func(y_val_clean, y_probs_val, metric_subfolder)
    perform_shap_analysis(booster_best, X_train_final, X_val_imp, y_val_clean, metric_subfolder)

    # Plot and save a normalized confusion matrix.
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm_norm, annot=True, fmt=".3f", cmap='Blues',
                xticklabels=['Predicted BBB-', 'Predicted BBB+'],
                yticklabels=['Actual BBB-', 'Actual BBB+'])
    plt.title('Normalised Confusion Matrix (Best Trial)')
    plt.tight_layout()
    plt.savefig(os.path.join(metric_subfolder, 'normalised_confusion_matrix_best.png'), dpi=300)
    plt.close()

    # Save the final best model.
    save_model(booster_best, metric_subfolder, model_name='final_model.json')
    
    # Save the feature names used in the final training set.
    feature_names = X_train_final.columns.tolist()
    joblib.dump(feature_names, os.path.join(metric_subfolder, "feature_names.joblib"))

    # Return the study object and final evaluation metrics.
    return study, final_metrics


def main():
    """
    Main pipeline function for training and optimizing the XGBoost model.
    
    This function:
      - Loads and preprocesses molecular data.
      - Accepts user-defined or command-line arguments for flexible configuration.
      - Supports both default model training and hyperparameter optimization via Optuna.
      - Handles data balancing, feature extraction (including fragment vectorization), and cross-validation.
      - Saves model outputs, evaluation metrics, and SHAP-based feature importance plots.
      
    Parameters:
        None (retrieves values from user input or command-line arguments).
    
    Returns:
        None (outputs results and saves files to the specified output directory).
    """
    
    parser = argparse.ArgumentParser(description="XGBoost BBB+/- Classification Script")
    parser.add_argument('--data_path', type=str, default=None, help='Path to input JSON data file')
    parser.add_argument('--output_dir', type=str, default=None, help='Directory to save outputs')
    parser.add_argument('--n_folds', type=int, default=None, help='Number of cross-validation folds')
    parser.add_argument('--random_seed', type=int, default=None, help='Random seed')
    parser.add_argument('--train_mode', type=str, default=None, help='Train mode: 1=Default, 2=Optimize')
    parser.add_argument('--balance_choice', type=str, default=None, help='Balancing method')
    parser.add_argument('--opt_metric', type=str, default=None, help='Which metric to optimize')
    parser.add_argument('--opt_trials', type=int, default=None, help='Number of Optuna trials')
    parser.add_argument('--use_gpu', type=str, default=None, help='Use GPU? (y/n)')
    args = parser.parse_args()

    # Get the data file path
    if not args.data_path:
        data_path = input("Enter the directory and name of training data file (.json): ").strip()
    else:
        data_path = args.data_path

    # Get the output directory
    if not args.output_dir:
        output_dir = input("Enter the directory to save outputs: ").strip()
    else:
        output_dir = args.output_dir

    # Determine number of folds for cross-validation
    if not args.n_folds:
        n_folds_input = input("Enter the number of cross-validation folds (default: 10): ").strip()
        n_folds = int(n_folds_input) if n_folds_input.isdigit() else DEFAULT_N_FOLDS
    else:
        n_folds = args.n_folds

    # Determine random seed for reproducibility
    if not args.random_seed:
        random_seed_input = input("Enter the random seed (default: 42): ").strip()
        random_seed = int(random_seed_input) if random_seed_input.isdigit() else DEFAULT_RANDOM_SEED
    else:
        random_seed = args.random_seed

    # Choose the training mode: default training or optimization
    if not args.train_mode:
        train_mode = input("Choose training mode:\n  (1) Default Model\n  (2) Optimize Model\n(Enter 1 or 2): ").strip()
    else:
        train_mode = args.train_mode

    # If optimization mode, determine the metric and number of trials
    if train_mode == '2':
        if not args.opt_metric:
            optimize_for_input = input(
                "Select metric to optimize:\n"
                "  (1) F1\n"
                "  (2) Balanced Accuracy\n"
                "  (3) Precision\n"
                "  (4) Recall\n"
                "  (5) Accuracy\n"
                "  (6) AUC\n"
                "  (all) for all\n"
                "(Enter choice): "
            ).strip()
        else:
            optimize_for_input = args.opt_metric

        metric_map = {
            '1': 'f1',
            '2': 'balanced accuracy',
            '3': 'precision',
            '4': 'recall',
            '5': 'accuracy',
            '6': 'auc'
        }
        if optimize_for_input.lower() in ['a', 'all']:
            optimize_for_list = ['f1', 'balanced accuracy', 'precision', 'recall', 'accuracy', 'auc']
        elif optimize_for_input in metric_map:
            optimize_for_list = [metric_map[optimize_for_input]]
        else:
            optimize_for_list = [optimize_for_input.lower()]

        if not args.opt_trials:
            trials_input = input("Number of Optuna trials (1-1000, default: 100): ").strip()
            num_trials = int(trials_input) if trials_input.isdigit() else 100
        else:
            num_trials = args.opt_trials
    else:
        optimize_for_list = ['balanced accuracy']
        num_trials = 0

    # Choose the data balancing method
    if not args.balance_choice:
        print("Choose data balancing method:\n  (1) None\n  (2) SMOTE\n  (3) SMOTEENN\n  (4) SMOTETomek")
        balance_choice_inp = input("Enter 1,2,3,4: ").strip()
    else:
        balance_choice_inp = args.balance_choice

    balance_map = {
        '1': None,
        '2': 'SMOTE',
        '3': 'SMOTEENN',
        '4': 'SMOTETomek'
    }
    balance_method = balance_map.get(balance_choice_inp, None)

    # Determine GPU usage for XGBoost configuration
    if not args.use_gpu:
        gpu_choice = input("Use GPU config? (y/n, default: y): ").strip().lower()
    else:
        gpu_choice = args.use_gpu.lower()

    if gpu_choice == 'n':
        tree_method = 'hist'
        predictor = 'cpu_predictor'
    else:
        tree_method = 'gpu_hist'
        predictor = 'gpu_predictor'

    # Create output directory and set up logging
    os.makedirs(output_dir, exist_ok=True)
    print(f"Output directory: {output_dir}")
    log_file_path = os.path.join(output_dir, 'model_training.log')
    setup_logging(log_file_path)
    logging.info("Started model training pipeline")

    # Load and preprocess data
    data = load_data(data_path)
    df = preprocess_data(data)
    df = handle_missing_values(df)
    df = clean_labels(df)
    cl_counts = df['BBB'].value_counts()
    logging.info(f"Class distribution:\n{cl_counts}")
    if len(cl_counts) < 2:
        logging.error("Only one class remains. Exiting.")
        sys.exit("Critical error: One class remains.")

    # Encode class labels and save the encoder for future use
    df, label_encoder = encode_labels(df)
    joblib.dump(label_encoder, os.path.join(output_dir, 'label_encoder.joblib'))

    # Create or load validation split
    val_data_path = os.path.join(output_dir, "validation_data.json")
    if not os.path.exists(val_data_path):
        y_all = df['BBB_Label']
        df_indices = df.index
        X_tr_idx, X_val_idx, _, _ = train_test_split(
            df_indices, y_all, test_size=0.2, random_state=random_seed, stratify=y_all
        )
        val_NO_list = df.loc[X_val_idx, 'NO.'].values.tolist()
        validation_data = {
            "train_idx": list(X_tr_idx),
            "val_idx": list(X_val_idx),
            "NO_list": val_NO_list
        }
        with open(val_data_path, "w") as f:
            json.dump(validation_data, f, indent=2)

        val_data_original = []
        set_val_no = set(val_NO_list)
        for entry in data:
            if '_preface' in entry:
                continue
            if 'NO.' in entry and entry['NO.'] in set_val_no:
                val_data_original.append(entry)
        with open(os.path.join(output_dir, "validation_data_original.json"), "w") as f:
            json.dump(val_data_original, f, indent=2)
    else:
        with open(val_data_path, 'r') as f:
            loaded_val = json.load(f)
        if 'train_idx' not in loaded_val or 'val_idx' not in loaded_val:
            y_all = df['BBB_Label']
            df_indices = df.index
            X_tr_idx, X_val_idx, _, _ = train_test_split(
                df_indices, y_all, test_size=0.2, random_state=random_seed, stratify=y_all
            )
            val_NO_list = df.loc[X_val_idx, 'NO.'].values.tolist()
            validation_data = {
                "train_idx": list(X_tr_idx),
                "val_idx": list(X_val_idx),
                "NO_list": val_NO_list
            }
            with open(val_data_path, "w") as f:
                json.dump(validation_data, f, indent=2)
        else:
            X_tr_idx = loaded_val["train_idx"]
            X_val_idx = loaded_val["val_idx"]

    # Split the data into training and validation sets
    df_train = df.loc[X_tr_idx].copy()
    df_val = df.loc[X_val_idx].copy()
    df_val = df_val.dropna(subset=['BBB_Label'])
    if df_val.empty:
        sys.exit("Critical error: Validation set empty after dropping BBB_Label=NaN.")

    # Vectorize molecular fragments using training data only
    dict_of_vects = vectorize_fragments_train_only(df_train)

    # If running in default training mode
    if train_mode == '1':
        # Retrieve vectorizers and frequency information
        brics_vect, brics_freq = dict_of_vects['BRICS']
        rings_vect, rings_freq = dict_of_vects['RINGS']
        side_vect, side_freq = dict_of_vects['SIDECHAINS']

        # Determine tokens to keep based on minimum frequency thresholds
        brics_keep = brics_freq[brics_freq >= MINIMUM_FRAGMENT_FREQUENCY[1]].index.tolist()
        rings_keep = rings_freq[rings_freq >= MINIMUM_FRAGMENT_FREQUENCY[2]].index.tolist()
        side_keep = side_freq[side_freq >= MINIMUM_FRAGMENT_FREQUENCY[3]].index.tolist()

        # Build training matrix by applying fragment vectorization and filtering tokens
        brics_df_tr = apply_fragments(df_train['BRIC_SMILES'], brics_vect, 'BRICS_')
        brics_df_tr = brics_df_tr[[c for c in brics_df_tr.columns if c.replace("BRICS_", "") in brics_keep]]
        rings_df_tr = apply_fragments(df_train['RINGS_SMILES'], rings_vect, 'RINGS_')
        rings_df_tr = rings_df_tr[[c for c in rings_df_tr.columns if c.replace("RINGS_", "") in rings_keep]]
        side_df_tr = apply_fragments(df_train['SIDE_CHAINS_SMILES'], side_vect, 'SIDECHAINS_')
        side_df_tr = side_df_tr[[c for c in side_df_tr.columns if c.replace("SIDECHAINS_", "") in side_keep]]

        # Get numeric features and combine with fragment features for training
        numeric_tr = df_train.drop(columns=[
            'NO.', 'BBB', 'LogBB', 'Group', 'BBB_Label',
            'BRIC_SMILES', 'RINGS_SMILES', 'SIDE_CHAINS_SMILES', 'SMILES'
        ], errors='ignore')
        X_train_base = pd.concat([numeric_tr, brics_df_tr, rings_df_tr, side_df_tr], axis=1)
        X_train_base = X_train_base.loc[:, ~X_train_base.columns.duplicated()].copy()
        X_train_base = X_train_base.apply(pd.to_numeric, errors='coerce')
        y_train_base = df_train['BBB_Label']

        # Save the training matrix for record keeping
        X_train_csv = os.path.join(output_dir, "train_matrix.csv")
        X_train_base.to_csv(X_train_csv, index=True)

        # Build validation matrix in similar fashion
        brics_df_val = apply_fragments(df_val['BRIC_SMILES'], brics_vect, 'BRICS_')
        brics_df_val = brics_df_val[[c for c in brics_df_val.columns if c.replace("BRICS_", "") in brics_keep]]
        rings_df_val = apply_fragments(df_val['RINGS_SMILES'], rings_vect, 'RINGS_')
        rings_df_val = rings_df_val[[c for c in rings_df_val.columns if c.replace("RINGS_", "") in rings_keep]]
        side_df_val = apply_fragments(df_val['SIDE_CHAINS_SMILES'], side_vect, 'SIDECHAINS_')
        side_df_val = side_df_val[[c for c in side_df_val.columns if c.replace("SIDECHAINS_", "") in side_keep]]

        numeric_val = df_val.drop(columns=[
            'NO.', 'BBB', 'LogBB', 'Group', 'BBB_Label',
            'BRIC_SMILES', 'RINGS_SMILES', 'SIDE_CHAINS_SMILES', 'SMILES'
        ], errors='ignore')
        X_val_base = pd.concat([numeric_val, brics_df_val, rings_df_val, side_df_val], axis=1)
        X_val_base = X_val_base.loc[:, ~X_val_base.columns.duplicated()].copy()
        X_val_base = X_val_base.apply(pd.to_numeric, errors='coerce')
        y_val_base = df_val['BBB_Label']

        # Save the validation matrix
        X_val_csv = os.path.join(output_dir, "val_matrix.csv")
        X_val_base.to_csv(X_val_csv, index=True)

        # Impute missing values in training and validation matrices
        X_train_imp, X_val_imp = impute_missing_values(X_train_base, X_val_base)

        # Apply data balancing if specified
        pos_before = (y_train_base == 1).sum()
        neg_before = (y_train_base == 0).sum()
        if balance_method is not None:
            if balance_method.lower() == 'smote':
                sampler = SMOTE(random_state=random_seed)
            elif balance_method.lower() == 'smoteenn':
                sampler = SMOTEENN(random_state=random_seed)
            elif balance_method.lower() == 'smotetomek':
                sampler = SMOTETomek(random_state=random_seed)
            else:
                sampler = None
        else:
            sampler = None

        if sampler is not None:
            X_train_bal, y_train_bal = sampler.fit_resample(X_train_imp, y_train_base)
            pos_after = (y_train_bal == 1).sum()
            neg_after = (y_train_bal == 0).sum()
        else:
            X_train_bal, y_train_bal = X_train_imp, y_train_base
            pos_after = pos_before
            neg_after = neg_before

        # Print a summary table of data balancing results
        print("\n=== Data Balancing Summary (Default Model) ===")
        print(f"{'Class':<10}{'Before':<15}{'After':<15}")
        print(f"{'BBB-':<10}{neg_before:<15}{neg_after:<15}")
        print(f"{'BBB+':<10}{pos_before:<15}{pos_after:<15}")

        pos_count = (y_train_bal == 1).sum()
        neg_count = (y_train_bal == 0).sum()
        scale_pos_weight_val = neg_count / pos_count if pos_count > 0 else 1.0

        # Perform cross-validation if number of folds > 1
        if n_folds > 1:
            skf_cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_seed)
            fold_metrics_list = []
            fold_idx = 1
            for train_index_fold, test_index_fold in skf_cv.split(X_train_bal, y_train_bal):
                X_tr_fold = X_train_bal.iloc[train_index_fold].copy()
                y_tr_fold = y_train_bal.iloc[train_index_fold]
                X_te_fold = X_train_bal.iloc[test_index_fold].copy()
                y_te_fold = y_train_bal.iloc[test_index_fold]

                booster_cv, _ = train_model(
                    X_tr_fold, y_tr_fold,
                    X_te_fold, y_te_fold,
                    random_seed=random_seed,
                    tree_method=tree_method,
                    predictor=predictor,
                    scale_pos_weight=scale_pos_weight_val,
                    num_boost_round=300,
                    early_stopping_rounds=20
                )
                fold_val_metrics, _, _ = evaluate_model(booster_cv, X_te_fold, y_te_fold, threshold=0.5)
                fold_val_metrics['fold'] = fold_idx
                fold_metrics_list.append(fold_val_metrics)
                metrics_str = ', '.join(f"{k}: {v}" for k, v in fold_val_metrics.items() if k != 'fold')
                print(f"[Fold {fold_idx}] metrics: {metrics_str}")
                fold_idx += 1

            cv_results_df = pd.DataFrame(fold_metrics_list)
            cv_results_path = os.path.join(output_dir, 'cross_validation_results.csv')
            cv_results_df.to_csv(cv_results_path, index=False)
            mean_metrics = cv_results_df.mean(numeric_only=True).to_dict()
            metrics_str = ', '.join(f"{k}: {v}" for k, v in mean_metrics.items() if k != 'fold')
            print(f"Average CV metrics across {n_folds} folds: {metrics_str}")
            logging.info(f"K-fold cross-validation saved to {cv_results_path}")

        # Final training of the default model using predefined parameters
        booster, _ = train_model(
            X_train_bal,
            y_train_bal,
            X_val_imp,
            y_val_base,
            random_seed=random_seed,
            tree_method=tree_method,
            predictor=predictor,
            scale_pos_weight=scale_pos_weight_val,
            num_boost_round=1418,
            early_stopping_rounds=50,
            **PREDEFINED_PARAMS
        )

        # Determine the best classification threshold based on balanced accuracy
        d_train_b = xgb.DMatrix(sanitize_column_names(X_train_bal))
        train_probs = booster.predict(d_train_b)
        y_train_array = y_train_bal.values.astype(int)
        best_threshold = 0.5
        best_score = -1
        for th in np.linspace(0.01, 0.99, 99):
            th_clamped = min(max(th, 0.35), 0.65)
            th_pred = (train_probs >= th_clamped).astype(int)
            score = balanced_accuracy_score(y_train_array, th_pred)
            if score > best_score:
                best_score = score
                best_threshold = th_clamped

        # Evaluate the final default model on the validation set
        val_metrics, y_probs_val, y_pred_val = evaluate_model(booster, X_val_imp, y_val_base, threshold=best_threshold)
        pd.DataFrame([val_metrics]).to_csv(os.path.join(output_dir, 'validation_metrics_initial.csv'), index=False)
        cm = confusion_matrix(y_val_base, y_pred_val, labels=[0, 1])
        save_confusion_matrix(cm, os.path.join(output_dir, 'validation_confusion_matrix_initial.csv'))

        with open(os.path.join(output_dir, 'best_threshold.txt'), 'w') as f:
            f.write(str(best_threshold))

        plot_roc_curve(y_val_base, y_probs_val, output_dir)
        plot_calibration_curve_func(y_val_base, y_probs_val, output_dir)
        perform_shap_analysis(booster, X_train_bal, X_val_imp, y_val_base, output_dir)

        cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        plt.figure(figsize=(6, 5))
        sns.heatmap(cm_norm, annot=True, fmt=".3f", cmap='Blues',
                    xticklabels=['Predicted BBB-', 'Predicted BBB+'],
                    yticklabels=['Actual BBB-', 'Actual BBB+'])
        plt.title('Normalised Confusion Matrix')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'normalised_confusion_matrix.png'), dpi=300)
        plt.close()

        save_model(booster, output_dir, model_name='xgboost_model.json')
        feature_names = X_train_bal.columns.tolist()
        joblib.dump(feature_names, os.path.join(output_dir, 'feature_names.joblib'))

        # Save final vectorizers and token lists for reproducibility
        joblib.dump(dict_of_vects['BRICS'][0], os.path.join(output_dir, "vectorizer_brics.joblib"))
        joblib.dump(dict_of_vects['RINGS'][0], os.path.join(output_dir, "vectorizer_rings.joblib"))
        joblib.dump(dict_of_vects['SIDECHAINS'][0], os.path.join(output_dir, "vectorizer_sidechains.joblib"))

        joblib.dump(brics_keep, os.path.join(output_dir, "kept_tokens_brics.joblib"))
        joblib.dump(rings_keep, os.path.join(output_dir, "kept_tokens_rings.joblib"))
        joblib.dump(side_keep, os.path.join(output_dir, "kept_tokens_sidechains.joblib"))

        print("\nFinal Model (Default) Training Completed Successfully.")
        print("Validation Set Performance Metrics:")
        for k, v in val_metrics.items():
            print(f"  {k}: {v}")

    # If running in optimization mode
    else:
        best_models_summary = {}
        y_train_series = df_train['BBB_Label']

        for metric in optimize_for_list:
            print(f"\nRunning Optuna optimization for {metric}...")
            study, final_metrics = run_optuna_optimization(
                df_train=df_train,
                y_train=y_train_series,
                df_val=df_val.copy(),
                dict_of_vectorizers=dict_of_vects,
                optimize_metric=metric,
                n_trials=num_trials,
                tree_method=tree_method,
                predictor=predictor,
                base_output_dir=output_dir,
                n_folds=n_folds,
                balance_method=balance_method,
                random_seed_main=random_seed
            )

            best_val_repr = f"{study.best_value:.4f}" if study.best_value is not None else "N/A"
            best_params = study.best_params
            print(f"Best trial for {metric}: {best_val_repr}")
            print("Best parameters:")
            for kk, vv in best_params.items():
                print(f"  {kk}: {vv}")

            best_dir = os.path.join(output_dir, metric.replace(" ", "_").lower())
            df_study = study.trials_dataframe(attrs=('number','values','value','params','state','user_attrs'))
            df_study.to_csv(os.path.join(best_dir, 'optuna_trials_raw.csv'), index=False)

            with open(os.path.join(best_dir, 'optuna_summary.txt'), 'w') as f:
                f.write(f"Best trial for {metric}: {best_val_repr}\n")
                f.write("Best params:\n")
                for p_k, p_v in best_params.items():
                    f.write(f"{p_k}: {p_v}\n")

            best_models_summary[metric] = {
                'best_study_value': best_val_repr,
                'best_params': best_params,
                'final_validation_metrics': final_metrics  # include all validation metrics
            }
            
            print("\nFinal Model Training and Evaluation Completed Successfully.")
            print("Validation Set Performance Metrics:")
            for kk, vv in final_metrics.items():
                print(f"  {kk}: {vv}")

        summary_file_path = os.path.join(output_dir, 'best_overall_models.txt')
        with open(summary_file_path, 'w') as f:
            f.write("Summary of Best Models for Each Metric:\n\n")
            for met, info in best_models_summary.items():
                f.write(f"=== Best {met.upper()} ===\n")
                f.write(f"  Optuna Best Value: {info['best_study_value']}\n")
                f.write("  Best Params:\n")
                for kk, vv in info['best_params'].items():
                    f.write(f"    {kk}: {vv}\n")
                f.write("  Final Validation Metrics:\n")
                for mk, mv in info['final_validation_metrics'].items():
                    f.write(f"    {mk}: {mv}\n")
                f.write("\n")
        print(f"\nA summary of all best models has been written to: {summary_file_path}")

    print("\nModel Training Pipeline Completed. Exiting.")


if __name__ == "__main__":
    main()
