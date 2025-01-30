"""
Author: Ben Franey
Version: 11.1.6 - Publish: 1.0
Last Review Date: 30-01-2025
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
  - Vectorizes molecular fragment information using 'CountVectorizer'.
  - Handles missing values using KNN imputation.
  - Supports multiple data balancing methods (None, SMOTE, SMOTEENN, SMOTETomek).

Training and Optimization:
  - Trains an XGBoost model with predefined hyperparameters.
  - Supports Optuna-based hyperparameter optimization with cross-validation.
  - Allows metric-based optimization (F1, AUC, accuracy, recall, precision, etc.).

Model Evaluation:
  - Performs cross-validation with key performance metrics.
  - Generates ROC curves, calibration curves, and confusion matrices.
  - Computes SHAP (SHapley Additive exPlanations) values for feature importance analysis.
  - Produces feature correlation heatmaps and importance rankings.

Results and Output:
  - Saves trained models and Optuna optimization logs.
  - Outputs validation set performance metrics.
  - Stores processed feature data and trained vectorizers for future use.

This script can be run with command-line arguments or interactively by prompting 
the user for inputs. It supports GPU acceleration for faster training.
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
MINIMUM_FRAGMENT_FREQUENCY = [5,5,5,5] # default, BRIC, RINGS, SIDECHAINS
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


# Clean output
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
        
        # Add row data for each molecule
        row_data = {
            'NO.': entry.get('NO.', np.nan),
            'BBB': entry.get('BBB+/BBB-', np.nan),
            'SMILES': entry.get('SMILES', np.nan),
            
            # Use PubChem only if exist then RDKit if not
            'LogP': entry.get('LogP_PubChem', np.nan) if not pd.isna(entry.get('LogP_PubChem', np.nan)) else entry.get('LogP_RDKit', np.nan),
            'Flexibility': entry.get('Flexibility_PubChem', np.nan) if not pd.isna(entry.get('Flexibility_PubChem', np.nan)) else entry.get('Flexibility_RDKit', np.nan),
            'HBA': entry.get('HBA_PubChem', np.nan) if not pd.isna(entry.get('HBA_PubChem', np.nan)) else entry.get('HBA_RDKit', np.nan),
            'HBD': entry.get('HBD_PubChem', np.nan) if not pd.isna(entry.get('HBD_PubChem', np.nan)) else entry.get('HBD_RDKit', np.nan),
            'TPSA': entry.get('TPSA_PubChem', np.nan) if not pd.isna(entry.get('TPSA_PubChem', np.nan)) else entry.get('TPSA_RDKit', np.nan),
            'Charge': entry.get('Charge_PubChem', np.nan) if not pd.isna(entry.get('Charge_PubChem', np.nan)) else entry.get('Charge_RDKit', np.nan),
            'Atom_Stereo': entry.get('AtomStereo_PubChem', np.nan) if not pd.isna(entry.get('AtomStereo_PubChem', np.nan)) else entry.get('AtomStereo_RDKit', np.nan),

            # More Descriptors
            'HeavyAtom': entry.get('HeavyAtom_RDKit', np.nan),
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
        
        # Collect SIDE CHAINS
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

    # Drop rows where BBB classification is missing
    df = df.dropna(subset=['BBB'])
    
    # Fill missing values in fragment-related text columns
    for col in ['BRIC_SMILES', 'RINGS', 'SIDE_CHAINS']:
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
        le = LabelEncoder()
        df['BBB_Label'] = le.fit_transform(df['BBB'])
        logging.info("Labels encoded successfully.")
        return df, le
    except Exception as e:
        logging.error(f"Error encoding labels: {e}")
        sys.exit(f"Critical error: {e}")


def vectorize_text_train(
    df, 
    text_col,
    min_freq=MINIMUM_FRAGMENT_FREQUENCY[0], 
    prefix=""
):
    """
    Trains a 'CountVectorizer' on a text column, filtering tokens based on frequency.

    - Uses 'CountVectorizer' to tokenize the text column ('text_col').
    - Retains only tokens appearing at least 'min_freq' times across all rows.
    - Optionally applies a prefix to column names for clarity.
    - Returns a DataFrame with token counts, the trained vectorizer, and the kept tokens.

    Parameters:
        df (pd.DataFrame): Input DataFrame containing the text column.
        text_col (str): Name of the column containing space-separated text tokens.
        min_freq (int, optional): Minimum occurrences for a token to be retained (default: MINIMUM_FRAGMENT_FREQUENCY[0]).
        prefix (str, optional): Prefix to apply to tokenized column names (default: "").

    Returns:
        tuple:
            pd.DataFrame: Token frequency matrix with filtered tokens.
            CountVectorizer: Trained CountVectorizer instance.
            list: List of retained tokens.

    Raises:
        SystemExit: If an error occurs, logs the error and exits the program.
    """
    try:
        # Create tokens
        vectorizer = CountVectorizer(token_pattern=r'[^ ]+', lowercase=False)
        text_features = vectorizer.fit_transform(df[text_col])
        feature_names = vectorizer.get_feature_names_out()
        temp_df = pd.DataFrame(text_features.toarray(), columns=feature_names, index=df.index)

        # Filter tokens based on minimum frequency
        freq = (temp_df > 0).sum(axis=0)
        keep_tokens = freq[freq >= min_freq].index.tolist()
        temp_df = temp_df[keep_tokens].fillna(0)

        # Apply prefix if provided
        if prefix:
            temp_df.columns = [f"{prefix}{c}" for c in temp_df.columns]

        return temp_df, vectorizer, keep_tokens
    
    except Exception as e:
        logging.error(f"Error in vectorize_text_train: {e}")
        sys.exit(f"Critical error: {e}")


def vectorize_text_apply(
    df, 
    text_col, 
    vectorizer, 
    keep_tokens, 
    prefix=""
):
    """
    Applies a trained 'CountVectorizer' to a text column, retaining only specified tokens.

    - Uses a pre-trained 'CountVectorizer' to transform 'df[text_col]' into token frequencies.
    - Retains only tokens specified in 'keep_tokens' to ensure consistent feature space.
    - Automatically adds missing columns (if any) with zero values to maintain alignment.
    - Optionally applies a prefix to the tokenized column names for clarity.

    Parameters:
        df (pd.DataFrame): Input DataFrame containing the text column.
        text_col (str): Column name containing space-separated text tokens.
        vectorizer (CountVectorizer): Pre-trained 'CountVectorizer' instance.
        keep_tokens (list): List of tokens to retain in the output.
        prefix (str, optional): Prefix to apply to column names (default: "").

    Returns:
        pd.DataFrame: Token frequency matrix with aligned columns.

    Raises:
        SystemExit: If an error occurs, logs the error and exits the program.
    """
    try:
        text_features = vectorizer.transform(df[text_col])
        feature_names = vectorizer.get_feature_names_out()
        temp_df = pd.DataFrame(text_features.toarray(), columns=feature_names, index=df.index)

        # Add missing columns to ensure alignment with training feature set
        missing = [tk for tk in keep_tokens if tk not in temp_df.columns]
        for mc in missing:
            temp_df[mc] = 0

        # Retain only the desired tokens, ensuring NaNs are replaced with zero
        temp_df = temp_df[keep_tokens].fillna(0)

        # Apply prefix if provided
        if prefix:
            temp_df.columns = [f"{prefix}{c}" for c in temp_df.columns]

        return temp_df
    
    except Exception as e:
        logging.error(f"Error in vectorize_text_apply: {e}")
        sys.exit(f"Critical error: {e}")


'''def impute_missing_values(X_train, X_val):
    """
    Performs KNN-based imputation on missing numeric values in training and validation datasets.

    - Aligns 'X_train' and 'X_val' by filling missing columns with zero to ensure compatibility.
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
        # Align columns between training and validation sets, filling missing ones with zeros
        X_train, X_val = X_train.align(X_val, join='outer', axis=1, fill_value=0)

        # Select numeric columns for imputation
        numeric_cols = list(X_train.select_dtypes(include=[np.number]).columns)
        combined = pd.concat([X_train, X_val], axis=0, ignore_index=True)

        # Apply KNN imputation if multiple numeric features exist
        if len(numeric_cols) > 1 and combined.shape[0] > 1:
            logging.info(f"KNN imputation on {len(numeric_cols)} numeric columns, total rows={combined.shape[0]}")
            knn_imputer = KNNImputer(n_neighbors=5)
            combined[numeric_cols] = knn_imputer.fit_transform(combined[numeric_cols])

        # Ensure no NaNs remain
        combined.fillna(0, inplace=True)

        # Split back into training and validation sets
        X_train_imputed = combined.iloc[:X_train.shape[0], :].copy()
        X_val_imputed = combined.iloc[X_train.shape[0]:, :].copy()

        # Final validation check for NaNs
        if X_train_imputed.isnull().sum().sum() > 0 or X_val_imputed.isnull().sum().sum() > 0:
            logging.error("NaNs remain after final KNN.")
            sys.exit("Critical error: NaNs remain after imputation.")
        else:
            logging.info("All missing values imputed successfully.")

        return X_train_imputed, X_val_imputed

    except Exception as e:
        logging.error(f"Error during KNN imputation: {e}")
        sys.exit(f"Critical error: {e}")'''


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
        # Check if there is more than one unique class in y_val
        if len(np.unique(y_val)) > 1:
            # Compute ROC curve and AUC score
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
    X_train,
    y_train,
    optimize_metric,
    n_trials,
    random_seed,
    tree_method,
    predictor,
    scale_pos_weight,
    output_dir,
    n_folds=DEFAULT_N_FOLDS
):
    """
    Run Optuna optimization for XGBoost with cross-validation.

    - Performs Bayesian optimization using Optuna to find optimal hyperparameters.
    - Uses Stratified K-Fold cross-validation for robust model evaluation.
    - Saves trial-specific models and logs optimization metrics.

    Parameters:
        X_train (pd.DataFrame): Training feature set.
        y_train (pd.Series): Training labels.
        optimize_metric (str): Metric to optimize ('AUC', 'F1', etc.).
        n_trials (int): Number of trials for Optuna optimization.
        random_seed (int): Random seed for reproducibility.
        tree_method (str): XGBoost tree-building method ('gpu_hist' or 'hist').
        predictor (str): XGBoost predictor type ('gpu_predictor' or 'cpu_predictor').
        scale_pos_weight (float): Weighting for imbalanced classification.
        output_dir (str): Directory for saving results and models.
        n_folds (int): Number of Stratified K-Fold splits (default: 'DEFAULT_N_FOLDS').

    Returns:
        optuna.Study: Optuna study object containing the optimization results.
    """

    # Ensure column names are compatible with XGBoost
    X_train = sanitize_column_names(X_train.copy())
    
    # Create directory for storing models
    model_subfolder = os.path.join(output_dir, f'all_models_{optimize_metric.replace(" ", "_")}')
    os.makedirs(model_subfolder, exist_ok=True)
    
    # Initialize Optuna study to maximize the selected metric
    study = optuna.create_study(direction='maximize')
    
    # Set up Stratified K-Fold cross-validation
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_seed)
    
    def objective(trial):
        """
        Objective function for Optuna to optimize XGBoost hyperparameters.

        - Performs cross-validation with selected hyperparameters.
        - Evaluates model performance on validation folds.
        - Stores trained models and metrics for each trial.

        Parameters:
            trial (optuna.Trial): Optuna trial object.

        Returns:
            float: Mean performance metric across folds.
        """
        
        # Define hyperparameter search space
        num_boost_round = trial.suggest_int('num_boost_round', 100, 3500)
        early_stopping_r = trial.suggest_int('early_stopping_rounds', 10, 400)
        max_depth = trial.suggest_int('max_depth', 6, 30)
        learning_rate = trial.suggest_float('learning_rate', 0.005, 0.1, log=True)
        subsample = trial.suggest_float('subsample', 0.4, 1.0)
        colsample_bytree = trial.suggest_float('colsample_bytree', 0.5, 1.0)
        min_child_weight = trial.suggest_int('min_child_weight', 1, 10)
        gamma = trial.suggest_float('gamma', 0.0, 5.0)
        reg_alpha = trial.suggest_float('reg_alpha', 0.0, 5.0)
        reg_lambda = trial.suggest_float('reg_lambda', 0.0, 5.0)
        
        # Define XGBoost parameters
        params = {
            'objective': 'binary:logistic',
            'eval_metric': 'auc',
            'seed': random_seed,
            'tree_method': tree_method,
            'predictor': predictor,
            'nthread': -1,
            'scale_pos_weight': scale_pos_weight,
            'max_depth': max_depth,
            'learning_rate': learning_rate,
            'subsample': subsample,
            'colsample_bytree': colsample_bytree,
            'min_child_weight': min_child_weight,
            'gamma': gamma,
            'alpha': reg_alpha,
            'lambda': reg_lambda
        }
        
        # Store performance metrics for each fold
        metrics_list = []
        
        # Perform Stratified K-Fold cross-validation
        for fold, (train_idx, valid_idx) in enumerate(skf.split(X_train, y_train), 1):
            X_tr, X_val_fold = X_train.iloc[train_idx], X_train.iloc[valid_idx]
            y_tr, y_val_fold = y_train.iloc[train_idx], y_train.iloc[valid_idx]
            
            # Create XGBoost DMatrix objects
            dtrain = xgb.DMatrix(X_tr, label=y_tr)
            dval = xgb.DMatrix(X_val_fold, label=y_val_fold)
            watchlist = [(dtrain, 'train'), (dval, 'eval')]
            
            # Train model with trial hyperparameters
            booster = xgb.train(
                params=params,
                dtrain=dtrain,
                num_boost_round=num_boost_round,
                evals=watchlist,
                early_stopping_rounds=early_stopping_r,
                verbose_eval=False
            )
            
            # Make predictions
            y_probs = booster.predict(dval)
            y_pred = (y_probs >= 0.5).astype(int)  # threshold can be adjusted later
            
            # Evaluate based on selected optimization metric
            if optimize_metric.lower() == 'f1':
                metric = f1_score(y_val_fold, y_pred)
            elif optimize_metric.lower() == 'balanced accuracy':
                metric = balanced_accuracy_score(y_val_fold, y_pred)
            elif optimize_metric.lower() == 'precision':
                metric = precision_score(y_val_fold, y_pred, zero_division=0)
            elif optimize_metric.lower() == 'recall':
                metric = recall_score(y_val_fold, y_pred, zero_division=0)
            elif optimize_metric.lower() == 'accuracy':
                metric = accuracy_score(y_val_fold, y_pred)
            elif optimize_metric.lower() == 'auc':
                if len(np.unique(y_val_fold)) > 1:
                    metric = roc_auc_score(y_val_fold, y_probs)
                else:
                    metric = 0.5  # default value when only one class is present
            else:
                raise ValueError(f"Unknown optimization metric '{optimize_metric}'")
            
            metrics_list.append(metric)
        # Compute mean performance metric across folds
        avg_metric = np.mean(metrics_list)
        
        # Store fold-wise metrics in Optuna trial attributes
        trial.set_user_attr("fold_metrics", metrics_list)
        # Store fold-wise metrics in Optuna trial attributes
        booster_filename = f"model_trial_{trial.number}.json"
        booster.save_model(os.path.join(model_subfolder, booster_filename))
        
        return avg_metric
    
    # Run Optuna optimization process
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    
    # Store results of all trials
    trial_metrics = []
    for t in study.trials:
        if t.state.name != "COMPLETE":
            continue
        d = {
            "trial_number": t.number,
            "value_for_optimization": t.value,
            "fold_metrics": t.user_attrs.get("fold_metrics", [])
        }
        trial_metrics.append(d)
    
    # Save trial metrics to CSV file
    df_all_metrics = pd.DataFrame(trial_metrics)
    csv_name = f'all_trials_metrics_{optimize_metric.replace(" ", "_")}.csv'
    df_all_metrics.to_csv(os.path.join(output_dir, csv_name), index=False)
    
    return study


def main():
    """
    Main pipeline function for training and optimizing the XGBoost model.

    - Loads and preprocesses molecular data.
    - Allows user-defined or command-line arguments for flexible configuration.
    - Supports both default training and hyperparameter optimization using Optuna.
    - Handles data balancing, feature extraction, and cross-validation.
    - Saves model outputs, evaluation metrics, and SHAP feature importance.

    Parameters:
        None (retrieves values from user input or command-line arguments).

    Returns:
        None (outputs results to the specified directory).
    """

    # Argument Parser
    parser = argparse.ArgumentParser(description="XGBoost BBB+/- Classification Script")
    parser.add_argument('--data_path', type=str, default=None, help='Path to input JSON data file')
    parser.add_argument('--output_dir', type=str, default=None, help='Directory to save outputs')
    parser.add_argument('--n_folds', type=int, default=None, help='Number of cross-validation folds')
    parser.add_argument('--random_seed', type=int, default=None, help='Random seed')
    parser.add_argument('--train_mode', type=str, default=None, help='Train mode (1: Default, 2: Optimise)')
    parser.add_argument('--balance_choice', type=str, default=None, help='Balancing method: 1=None,2=SMOTE,3=SMOTEENN,4=SMOTETomek')
    parser.add_argument('--opt_metric', type=str, default=None, help='Metric to optimise (f1, balanced accuracy, precision, recall, accuracy, auc, all)')
    parser.add_argument('--opt_trials', type=int, default=None, help='Number of Optuna trials, e.g. 100')
    parser.add_argument('--use_gpu', type=str, default=None, help='Use GPU config? (y or n)')

    args = parser.parse_args()

    # Fallback to user input if not provided via args
    if not args.data_path:
        data_path = input("Enter the directory and name of training data file (.json): ").strip()
    else:
        data_path = args.data_path

    if not args.output_dir:
        output_dir = input("Enter the directory to save outputs: ").strip()
    else:
        output_dir = args.output_dir

    if not args.n_folds:
        n_folds_input = input("Enter the number of cross-validation folds (default: 10): ").strip()
        n_folds = int(n_folds_input) if n_folds_input.isdigit() else DEFAULT_N_FOLDS
    else:
        n_folds = args.n_folds

    if not args.random_seed:
        random_seed_input = input("Enter the random seed (default: 42): ").strip()
        random_seed = int(random_seed_input) if random_seed_input.isdigit() else DEFAULT_RANDOM_SEED
    else:
        random_seed = args.random_seed

    if not args.train_mode:
        train_mode = input("Choose training mode:\n  (1) Default Model\n  (2) Optimize Model\n(Enter 1 or 2): ").strip()
    else:
        train_mode = args.train_mode

    # Optomisation pathway
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

    if not args.balance_choice:
        print("Choose data balancing method:\n  (1) None\n  (2) SMOTE\n  (3) SMOTEENN\n  (4) SMOTETomek")
        balance_choice = input("Enter 1,2,3,4: ").strip()
    else:
        balance_choice = args.balance_choice

    balance_method = {
        '1': None,
        '2': 'SMOTE',
        '3': 'SMOTEENN',
        '4': 'SMOTETomek'
    }.get(balance_choice, None)

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

    os.makedirs(output_dir, exist_ok=True)
    print(f"Output directory: {output_dir}")
    log_file_path = os.path.join(output_dir, 'model_training.log')
    setup_logging(log_file_path)
    logging.info("Started model training pipeline")

    # Load and preprocess
    data = load_data(data_path)
    df = preprocess_data(data)
    df = handle_missing_values(df)
    df = clean_labels(df)

    cl_counts = df['BBB'].value_counts()
    logging.info(f"Class distribution after cleaning:\n{cl_counts}")
    if len(cl_counts) < 2:
        logging.error("Only one class remains. Exiting.")
        sys.exit("Critical error: One class remains.")

    df, label_encoder = encode_labels(df)
    joblib.dump(label_encoder, os.path.join(output_dir, 'label_encoder.joblib'))

    # CREATE OR USE EXISTING VAL SPLIT
    val_data_path = os.path.join(output_dir, "validation_data.json")
    vectorizer_brics_path = os.path.join(output_dir, "vectorizer_brics.joblib")
    keep_tokens_brics_path = os.path.join(output_dir, "kept_tokens_brics.joblib")
    vectorizer_rings_path = os.path.join(output_dir, "vectorizer_rings.joblib")
    keep_tokens_rings_path = os.path.join(output_dir, "kept_tokens_rings.joblib")
    vectorizer_side_path = os.path.join(output_dir, "vectorizer_sidechains.joblib")
    keep_tokens_side_path = os.path.join(output_dir, "kept_tokens_sidechains.joblib")

    if not os.path.exists(val_data_path):
        # TRAIN VECTORIZERS
        bric_df, vectorizer_brics, keep_tokens_brics = vectorize_text_train(df, 'BRIC_SMILES', min_freq=MINIMUM_FRAGMENT_FREQUENCY[1], prefix="BRICS_")
        joblib.dump(vectorizer_brics, vectorizer_brics_path)
        joblib.dump(keep_tokens_brics, keep_tokens_brics_path)

        rings_df, vectorizer_rings, keep_tokens_rings = vectorize_text_train(df, 'RINGS_SMILES', min_freq=MINIMUM_FRAGMENT_FREQUENCY[2], prefix="RINGS_")
        joblib.dump(vectorizer_rings, vectorizer_rings_path)
        joblib.dump(keep_tokens_rings, keep_tokens_rings_path)

        side_df, vectorizer_side, keep_tokens_side = vectorize_text_train(df, 'SIDE_CHAINS_SMILES', min_freq=MINIMUM_FRAGMENT_FREQUENCY[3], prefix="SIDECHAINS_")
        joblib.dump(vectorizer_side, vectorizer_side_path)
        joblib.dump(keep_tokens_side, keep_tokens_side_path)

        X_all = pd.concat([
            df.drop(columns=['NO.', 'BBB', 'BBB_Label', 'LogBB', 'Group', 'BRIC_SMILES', 'RINGS_SMILES', 'SIDE_CHAINS_SMILES', 'SMILES'], errors='ignore'),
            bric_df,
            rings_df,
            side_df
        ], axis=1)

        X_all = X_all.apply(pd.to_numeric, errors='coerce')
        X_all = X_all.loc[:, ~X_all.columns.duplicated()].copy()

        y_all = df['BBB_Label']

        # Split
        X_train_full, X_val_full, y_train_full, y_val_full = train_test_split(
            X_all, y_all, test_size=0.2, random_state=random_seed, stratify=y_all
        )
        val_NO = df.loc[X_val_full.index, 'NO.'].values.tolist()

        validation_data = {
            "X": X_val_full.to_dict(orient="records"),
            "y": y_val_full.tolist(),
            "NO_list": val_NO
        }
        with open(val_data_path, "w") as f:
            json.dump(validation_data, f)

        val_data_original = []
        set_val_no = set(val_NO)
        for entry in data:
            if '_preface' in entry:
                continue
            if 'NO.' in entry and entry['NO.'] in set_val_no:
                val_data_original.append(entry)
        with open(os.path.join(output_dir, "validation_data_original.json"), "w") as f:
            json.dump(val_data_original, f, indent=2)

        df_train = df[~df['NO.'].isin(val_NO)].reset_index(drop=True)
        val_NO_list = val_NO
    else:
        with open(val_data_path, 'r') as f:
            val_data = json.load(f)
        X_val_dicts = val_data["X"]
        y_val_list = val_data["y"]
        val_NO_list = val_data.get("NO_list", [])

        vectorizer_brics = joblib.load(vectorizer_brics_path)
        keep_tokens_brics = joblib.load(keep_tokens_brics_path)
        vectorizer_rings = joblib.load(vectorizer_rings_path)
        keep_tokens_rings = joblib.load(keep_tokens_rings_path)
        vectorizer_side = joblib.load(vectorizer_side_path)
        keep_tokens_side = joblib.load(keep_tokens_side_path)

        X_val_full = pd.DataFrame(X_val_dicts)
        X_val_full = X_val_full.apply(pd.to_numeric, errors='coerce')
        X_val_full = X_val_full.loc[:, ~X_val_full.columns.duplicated()].copy()

        y_val_full = pd.Series(y_val_list)
        df_train = df[~df['NO.'].isin(val_NO_list)].reset_index(drop=True)

    # VECTORIZING FOR TRAIN/VAL
    drop_cols_train = ['NO.', 'BBB', 'BBB_Label', 'LogBB', 'Group', 'BRIC_SMILES', 'RINGS_SMILES', 'SIDE_CHAINS_SMILES', 'SMILES']
    df_train_brics = vectorize_text_apply(df_train, 'BRIC_SMILES', vectorizer_brics, keep_tokens_brics, prefix="BRICS_")
    df_train_rings = vectorize_text_apply(df_train, 'RINGS_SMILES', vectorizer_rings, keep_tokens_rings, prefix="RINGS_")
    df_train_side = vectorize_text_apply(df_train, 'SIDE_CHAINS_SMILES', vectorizer_side, keep_tokens_side, prefix="SIDECHAINS_")

    X_train = pd.concat([
        df_train.drop(columns=drop_cols_train, errors='ignore'),
        df_train_brics,
        df_train_rings,
        df_train_side
    ], axis=1)

    X_train = X_train.loc[:, ~X_train.columns.duplicated()].copy()
    X_train = X_train.apply(pd.to_numeric, errors='coerce')

    y_train = df_train['BBB_Label']

    df_val_subset = df[df['NO.'].isin(val_NO_list)].reset_index(drop=True)

    df_val_brics = vectorize_text_apply(df_val_subset, 'BRIC_SMILES', vectorizer_brics, keep_tokens_brics, prefix="BRICS_")
    df_val_rings = vectorize_text_apply(df_val_subset, 'RINGS_SMILES', vectorizer_rings, keep_tokens_rings, prefix="RINGS_")
    df_val_side = vectorize_text_apply(df_val_subset, 'SIDE_CHAINS_SMILES', vectorizer_side, keep_tokens_side, prefix="SIDECHAINS_")

    X_val = pd.concat([
        X_val_full.reset_index(drop=True),
        df_val_brics.reset_index(drop=True),
        df_val_rings.reset_index(drop=True),
        df_val_side.reset_index(drop=True)
    ], axis=1)

    X_val = X_val.loc[:, ~X_val.columns.duplicated()].copy()
    X_val = X_val.apply(pd.to_numeric, errors='coerce')

    y_val = pd.Series(y_val_full.values, index=X_val.index)

    # IMPUTE
    X_train_imputed, X_val_imputed = impute_missing_values(X_train, X_val)
    X_val_imputed.reset_index(drop=True, inplace=True)
    y_train.reset_index(drop=True, inplace=True)
    y_val.reset_index(drop=True, inplace=True)

    if not all(pd.api.types.is_numeric_dtype(dtype) for dtype in X_train_imputed.dtypes):
        logging.error("Non-numeric columns detected in X_train_imputed.")
        sys.exit("Critical error: Non-numeric columns detected in X_train_imputed.")
    if not all(pd.api.types.is_numeric_dtype(dtype) for dtype in X_val_imputed.dtypes):
        logging.error("Non-numeric columns detected in X_val_imputed.")
        sys.exit("Critical error: Non-numeric columns detected in X_val_imputed.")

    # BALANCING
    original_counts = y_train.value_counts().to_dict()
    if balance_method is not None:
        if balance_method == 'SMOTE':
            sampler = SMOTE(random_state=random_seed)
        elif balance_method == 'SMOTEENN':
            sampler = SMOTEENN(random_state=random_seed)
        elif balance_method == 'SMOTETomek':
            sampler = SMOTETomek(random_state=random_seed)
        else:
            sampler = None

        if sampler is not None:
            X_train_balanced, y_train_balanced = sampler.fit_resample(X_train_imputed, y_train)
            bal_counts = y_train_balanced.value_counts().to_dict()
            print("\nData Balancing Summary:")
            print(f"{'Class':<10}{'Before Balancing':<22}{'After Balancing':<22}{'Net Change':<12}")
            for clsval in original_counts.keys():
                lab = "BBB+" if clsval == 1 else "BBB-"
                before_count = original_counts.get(clsval, 0)
                after_count = bal_counts.get(clsval, 0)
                net = after_count - before_count 
                print(f"{lab:<10}{before_count:<22}{after_count:<22}{net:<12}")

        else:
            X_train_balanced, y_train_balanced = X_train_imputed, y_train
    else:
        print(f"Number of training samples: {len(y_train)}")
        X_train_balanced, y_train_balanced = X_train_imputed, y_train
        
    # Save the processed training features for KNN imputation in prediction
    processed_train_features_path = os.path.join(output_dir, 'processed_train_features.csv')
    X_train_balanced.to_csv(processed_train_features_path, index=False)
    logging.info(f"Processed training features saved at {processed_train_features_path}")

    pos_count = (y_train_balanced == 1).sum()
    neg_count = (y_train_balanced == 0).sum()
    scale_pos_weight_value = neg_count / pos_count if pos_count > 0 else 1.0
    joblib.dump(scale_pos_weight_value, os.path.join(output_dir, 'scale_pos_weight.joblib'))

    # CROSS-VALIDATION
    if n_folds > 1:
        from sklearn.model_selection import StratifiedKFold
        print(f"\nPerforming {n_folds}-fold cross-validation on the training set...")
        skf_cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_seed)
        fold_metrics_list = []
        fold_idx = 1
        for train_index, val_index in skf_cv.split(X_train_balanced, y_train_balanced):
            X_tr_fold = X_train_balanced.iloc[train_index]
            y_tr_fold = y_train_balanced.iloc[train_index]
            X_val_fold = X_train_balanced.iloc[val_index]
            y_val_fold = y_train_balanced.iloc[val_index]

            booster_cv, _ = train_model(
                X_tr_fold,
                y_tr_fold,
                X_val_fold,
                y_val_fold,
                random_seed=random_seed,
                tree_method=tree_method,
                predictor=predictor,
                scale_pos_weight=scale_pos_weight_value,
                num_boost_round=300,
                early_stopping_rounds=20
            )

            fold_val_metrics, _, _ = evaluate_model(booster_cv, X_val_fold, y_val_fold, threshold=0.5)
            fold_val_metrics['fold'] = fold_idx
            fold_metrics_list.append(fold_val_metrics)
            
            metrics_str = ', '.join(f"{key}: {value}" for key, value in fold_val_metrics.items() if key != 'fold')
            print(f"[Fold {fold_idx}] metrics: {metrics_str}")
            fold_idx += 1

        cv_results_df = pd.DataFrame(fold_metrics_list)
        cv_results_path = os.path.join(output_dir, 'cross_validation_results.csv')
        cv_results_df.to_csv(cv_results_path, index=False)
        mean_metrics = cv_results_df.mean(numeric_only=True).to_dict()

        metrics_str = ', '.join(f"{key}: {value}" for key, value in mean_metrics.items() if key != 'fold')

        print(f"Average CV metrics across {n_folds} folds: {metrics_str}")
        logging.info(f"K-fold cross-validation results saved to: {cv_results_path}")
        logging.info(f"Average CV metrics across {n_folds} folds: {mean_metrics}")

    best_models_summary = {}

    # FINAL TRAINING / OPTUNA
    if train_mode == '2':
        import optuna
        for metric in optimize_for_list:
            metric_dir = metric.replace(' ', '_').lower()
            metric_output_dir = os.path.join(output_dir, metric_dir)
            os.makedirs(metric_output_dir, exist_ok=True)

            metric_log_file = os.path.join(metric_output_dir, 'model_training.log')
            setup_logging(metric_log_file)
            logging.info(f"Optuna optimization for {metric}")

            print(f"\nRunning Optuna optimization for {metric}...")
            study = run_optuna_optimization(
                X_train_balanced,
                y_train_balanced,
                optimize_metric=metric,
                n_trials=num_trials,
                random_seed=random_seed,
                tree_method=tree_method,
                predictor=predictor,
                scale_pos_weight=scale_pos_weight_value,
                output_dir=metric_output_dir,
                n_folds=n_folds
            )

            best_value_repr = str(study.best_value)
            best_params = study.best_params
            print(f"Best trial for {metric}: {best_value_repr}")
            print("Best parameters:")
            for k, v in best_params.items():
                print(f"  {k}: {v}")

            df_trials = study.trials_dataframe(attrs=('number','values','value','params','state'))
            df_trials.to_csv(os.path.join(metric_output_dir, 'optuna_trials_raw.csv'), index=False)

            with open(os.path.join(metric_output_dir, 'optuna_summary.txt'), 'w') as f:
                f.write(f"Best trial for {metric}: {best_value_repr}\n")
                f.write("Best params:\n")
                for kk, vv in best_params.items():
                    f.write(f"{kk}: {vv}\n")

            final_num_round = best_params.pop('num_boost_round', 1000)
            final_esr = best_params.pop('early_stopping_rounds', None)

            booster, evals_result = train_model(
                X_train_balanced,
                y_train_balanced,
                X_val_imputed,
                y_val,
                random_seed=random_seed,
                tree_method=tree_method,
                predictor=predictor,
                scale_pos_weight=scale_pos_weight_value,
                num_boost_round=final_num_round,
                early_stopping_rounds=final_esr,
                **best_params
            )

            # Find best threshold on training set for that metric
            d_train_b = xgb.DMatrix(sanitize_column_names(X_train_balanced.copy()))
            train_probs = booster.predict(d_train_b)
            y_train_array = y_train_balanced.values.astype(int)

            best_threshold = 0.5
            best_score = -1
            thresholds = np.linspace(0.01, 0.99, 99)
            for th in thresholds:
                th_clamped = min(max(th, 0.35), 0.65)
                th_pred = (train_probs >= th_clamped).astype(int)

                if metric == 'f1':
                    score = f1_score(y_train_array, th_pred)
                elif metric == 'balanced accuracy':
                    score = balanced_accuracy_score(y_train_array, th_pred)
                elif metric == 'precision':
                    score = precision_score(y_train_array, th_pred, zero_division=0)
                elif metric == 'recall':
                    score = recall_score(y_train_array, th_pred, zero_division=0)
                elif metric == 'accuracy':
                    score = accuracy_score(y_train_array, th_pred)
                elif metric == 'auc':
                    score = roc_auc_score(y_train_array, train_probs)
                else:
                    score = f1_score(y_train_array, th_pred)

                if score > best_score:
                    best_score = score
                    best_threshold = th_clamped

            val_metrics, y_probs_val, y_pred_val = evaluate_model(
                booster, X_val_imputed, y_val, threshold=best_threshold
            )
            pd.DataFrame([val_metrics]).to_csv(os.path.join(metric_output_dir, 'validation_metrics_initial.csv'), index=False)
            cm = confusion_matrix(y_val, y_pred_val, labels=[0, 1])
            save_confusion_matrix(cm, os.path.join(metric_output_dir, 'validation_confusion_matrix_initial.csv'))

            with open(os.path.join(metric_output_dir, 'best_threshold.txt'), 'w') as f:
                f.write(str(best_threshold))

            plot_roc_curve(y_val, y_probs_val, metric_output_dir)
            plot_calibration_curve_func(y_val, y_probs_val, metric_output_dir)
            perform_shap_analysis(booster, X_train_balanced, X_val_imputed, y_val, metric_output_dir)

            cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            plt.figure(figsize=(6, 5))
            sns.heatmap(cm_norm, annot=True, fmt=".3f", cmap='Blues',
                        xticklabels=['Predicted BBB-', 'Predicted BBB+'],
                        yticklabels=['Actual BBB-', 'Actual BBB+'])
            plt.title('Normalised Confusion Matrix')
            plt.tight_layout()
            plt.savefig(os.path.join(metric_output_dir, 'normalised_confusion_matrix.png'), dpi=300)
            plt.close()

            trial_values = [t.value for t in study.trials if t.value is not None and t.state.name == "COMPLETE"]
            trial_values_sorted = sorted(trial_values, reverse=True)
            plt.figure(figsize=(10, 6))
            plt.plot(trial_values_sorted, marker='o')
            plt.title(f'{metric.capitalize()} Values for All Trials (Sorted)')
            plt.xlabel('Trial Rank')
            plt.ylabel(metric.capitalize())
            plt.tight_layout()
            plt.savefig(os.path.join(metric_output_dir, 'pareto_chart.png'), dpi=300)
            plt.close()

            save_model(booster, metric_output_dir, model_name='final_model.json')
            feature_names = X_train_balanced.columns.tolist()
            joblib.dump(feature_names, os.path.join(metric_output_dir, 'feature_names.joblib'))

            best_models_summary[metric] = {
                'best_study_value': best_value_repr,
                'best_params': best_params,
                'validation_metrics': val_metrics
            }

            print("\nFinal Model Training and Evaluation Completed Successfully.")
            print("Validation Set Performance Metrics:")
            for kk, vv in val_metrics.items():
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
                for mk, mv in info['validation_metrics'].items():
                    f.write(f"    {mk}: {mv}\n")
                f.write("\n")
        print(f"\nA summary of all best models has been written to: {summary_file_path}")

    else:
        # Default training
        if y_val.isnull().any():
            logging.error("NaNs in y_val. Dropping them.")
            valid_mask = ~y_val.isnull()
            X_val_imputed = X_val_imputed[valid_mask]
            y_val = y_val[valid_mask]

        if y_val.empty:
            logging.error("Validation set empty after dropping NaNs.")
            sys.exit("Critical error: Validation empty.")

        booster, evals_result = train_model(
            X_train_balanced,
            y_train_balanced,
            X_val_imputed,
            y_val,
            random_seed=random_seed,
            tree_method=tree_method,
            predictor=predictor,
            scale_pos_weight=scale_pos_weight_value,
            num_boost_round=1418,
            early_stopping_rounds=50,
            **PREDEFINED_PARAMS
        )

        d_train_b = xgb.DMatrix(sanitize_column_names(X_train_balanced.copy()))
        train_probs = booster.predict(d_train_b)
        y_train_array = y_train_balanced.values.astype(int)

        best_threshold = 0.5
        best_score = -1
        thresholds = np.linspace(0.01, 0.99, 99)
        for th in thresholds:
            th_clamped = min(max(th, 0.35), 0.65)
            th_pred = (train_probs >= th_clamped).astype(int)
            score = balanced_accuracy_score(y_train_array, th_pred)
            if score > best_score:
                best_score = score
                best_threshold = th_clamped

        val_metrics, y_probs_val, y_pred_val = evaluate_model(booster, X_val_imputed, y_val, threshold=best_threshold)
        pd.DataFrame([val_metrics]).to_csv(os.path.join(output_dir, 'validation_metrics_initial.csv'), index=False)
        cm = confusion_matrix(y_val, y_pred_val, labels=[0, 1])
        save_confusion_matrix(cm, os.path.join(output_dir, 'validation_confusion_matrix_initial.csv'))

        with open(os.path.join(output_dir, 'best_threshold.txt'), 'w') as f:
            f.write(str(best_threshold))

        plot_roc_curve(y_val, y_probs_val, output_dir)
        plot_calibration_curve_func(y_val, y_probs_val, output_dir)
        perform_shap_analysis(booster, X_train_balanced, X_val_imputed, y_val, output_dir)

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
        feature_names = X_train_balanced.columns.tolist()
        joblib.dump(feature_names, os.path.join(output_dir, 'feature_names.joblib'))

        print("\nFinal Model Training and Evaluation Completed Successfully.")
        print("Validation Set Performance Metrics:")
        for k, v in val_metrics.items():
            print(f"  {k}: {v}")
    
    print("\nFinal Model Training and Evaluation Completed. Exiting.")


if __name__ == "__main__":
    main()
