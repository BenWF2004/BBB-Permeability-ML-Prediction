"""
Author: Ben Franey
Version: 11.1.2 - Publish: 1.0
Last Review Date: 30-01-2025
Overview:
    Predict BBB+/- labels for molecules using pre-trained XGBoost models.
    Adjusts mean and median probability calculations based on individual model thresholds.
    Outputs all model prediction probabilities and labels alongside aggregated binary labels.
"""

DEFAULT_KNN_NEIGHBOURS = 5  # For KNNImputer, originally 5

import os
import sys
import json
import logging
import warnings
import argparse

import numpy as np
import pandas as pd
import joblib
import xgboost as xgb

from tqdm import tqdm

from sklearn.impute import KNNImputer
from sklearn.feature_extraction.text import CountVectorizer

warnings.filterwarnings("ignore")


def setup_logging(log_file_path: str):
    """
    Configure logging to log messages to both a file and the console.
    
    Parameters:
        log_file_path (str): Path where log file should be saved.
    
    Returns:
        None
    """
    
    logging.basicConfig(
        filename=log_file_path,
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)


def load_data(input_json_path: str) -> list:
    """
    Load molecular data from a JSON file.
    
    Parameters:
        input_json_path (str): Path to the input JSON file.
    
    Returns:
        list: List of dictionaries containing molecular data.
    
    Raises:
        SystemExit: If the JSON file does not exist or cannot be loaded.
    """
    
    if not os.path.exists(input_json_path):
        sys.exit(f"Critical error: Input JSON file '{input_json_path}' does not exist.")

    try:
        with open(input_json_path, 'r') as f:
            data = json.load(f)
        if not isinstance(data, list):
            # If not a list of molecules, check for dictionary with 'molecules'
            if isinstance(data, dict):
                if 'molecules' in data and isinstance(data['molecules'], list):
                    data = data['molecules']
                else:
                    sys.exit("Critical error: JSON structure not recognised. Expecting a list or { 'molecules': [...] }.")
            else:
                sys.exit("Critical error: JSON structure not recognised. Expecting a list or { 'molecules': [...] }.")
        logging.info(f"Data loaded from '{input_json_path}', total records: {len(data)}")
        return data
    except Exception as e:
        logging.error(f"Error loading JSON data: {e}")
        sys.exit(f"Critical error: {e}")


def preprocess_data(raw_data: list) -> pd.DataFrame:
    """
    Convert JSON molecular data into a structured DataFrame.
    Extracts relevant molecular properties and fragments.
    
    Parameters:
        raw_data (list): List of dictionaries containing molecular data.
    
    Returns:
        pd.DataFrame: Processed DataFrame containing extracted molecular descriptors.
    """
    
    df_list = []
    for entry in tqdm(raw_data, desc="Preprocessing entries"):
        if '_preface' in entry:
            continue

        smiles = entry.get('SMILES', '')
        if not smiles:
            continue

        row_data = {
            'NO.': entry.get('NO.', np.nan),
            'BBB': entry.get('BBB+/BBB-', np.nan),
            'SMILES': entry.get('SMILES', np.nan),
            
            'LogP': entry.get('LogP_PubChem', np.nan) if not pd.isna(entry.get('LogP_PubChem', np.nan)) else entry.get('LogP_RDKit', np.nan),
            'Flexibility': entry.get('Flexibility_PubChem', np.nan) if not pd.isna(entry.get('Flexibility_PubChem', np.nan)) else entry.get('Flexibility_RDKit', np.nan),
            'HBA': entry.get('HBA_PubChem', np.nan) if not pd.isna(entry.get('HBA_PubChem', np.nan)) else entry.get('HBA_RDKit', np.nan),
            'HBD': entry.get('HBD_PubChem', np.nan) if not pd.isna(entry.get('HBD_PubChem', np.nan)) else entry.get('HBD_RDKit', np.nan),
            'TPSA': entry.get('TPSA_PubChem', np.nan) if not pd.isna(entry.get('TPSA_PubChem', np.nan)) else entry.get('TPSA_RDKit', np.nan),
            'Charge': entry.get('Charge_PubChem', np.nan) if not pd.isna(entry.get('Charge_PubChem', np.nan)) else entry.get('Charge_RDKit', np.nan),
            'Atom_Stereo': entry.get('AtomStereo_PubChem', np.nan) if not pd.isna(entry.get('AtomStereo_PubChem', np.nan)) else entry.get('AtomStereo_RDKit', np.nan),

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

    df = pd.DataFrame(df_list)
    logging.info(f"Preprocessed DataFrame shape: {df.shape}")
    return df


def handle_missing_bbb_column(df: pd.DataFrame) -> pd.DataFrame:
    """
    Handles missing 'BBB' column in the input DataFrame.

    - Checks if the 'BBB' column is present.
    - If missing, creates the column and fills it with NaN values.
    - Ensures the DataFrame structure remains intact.

    Parameters:
        df (pd.DataFrame): Input DataFrame.

    Returns:
        pd.DataFrame: DataFrame with the 'BBB' column ensured.
    """
    
    if 'BBB' not in df.columns:
        df['BBB'] = np.nan
    return df


def clean_bbb_labels(df: pd.DataFrame) -> pd.DataFrame:
    """
    Standardizes 'BBB' classification labels for consistency.

    - Converts 'BBB' column values to uppercase.
    - Strips excess whitespace from the labels.
    - Replaces variations like 'BBB +' and 'BBB -' with standardized 'BBB+' and 'BBB-'.
    - Ensures only 'BBB+' and 'BBB-' are retained.

    Parameters:
        df (pd.DataFrame): Input DataFrame containing 'BBB' labels.

    Returns:
        pd.DataFrame: DataFrame with standardized 'BBB' classification labels.
    """
    
    df['BBB'] = df['BBB'].astype(str).str.strip().str.upper()
    df['BBB'] = df['BBB'].replace({
        'BBB +': 'BBB+',
        'BBB -': 'BBB-',
        'BBB+': 'BBB+',
        'BBB-': 'BBB-'
    })
    return df


def vectorize_text_apply(df: pd.DataFrame, text_col: str, vectorizer: CountVectorizer, keep_tokens: list, prefix: str = "") -> pd.DataFrame:
    """
    Applies a trained `CountVectorizer` to transform a specified text column into token frequencies.

    - Uses a pre-trained `CountVectorizer` to transform `df[text_col]` into token frequencies.
    - Retains only tokens specified in `keep_tokens` to ensure consistency with training data.
    - Automatically adds missing columns (if any) with zero values to maintain feature alignment.
    - Optionally applies a prefix to the tokenized column names for better identification.

    Parameters:
        df (pd.DataFrame): Input DataFrame containing the text column.
        text_col (str): Column name containing space-separated text tokens.
        vectorizer (CountVectorizer): Pre-trained `CountVectorizer` instance.
        keep_tokens (list): List of tokens to retain in the output.
        prefix (str, optional): Prefix to apply to column names (default: "").

    Returns:
        pd.DataFrame: Token frequency matrix with aligned columns.

    Raises:
        SystemExit: If an error occurs, logs the error and exits the program.
    """
    
    if text_col not in df.columns:
        df[text_col] = ""
    try:
        text_features = vectorizer.transform(df[text_col])
        feature_names = vectorizer.get_feature_names_out()
        temp_df = pd.DataFrame(text_features.toarray(), columns=feature_names, index=df.index)

        missing = [tk for tk in keep_tokens if tk not in temp_df.columns]
        for mc in missing:
            temp_df[mc] = 0

        temp_df = temp_df[keep_tokens].fillna(0)

        if prefix:
            temp_df.columns = [f"{prefix}{c}" for c in temp_df.columns]

        return temp_df
    except Exception as e:
        logging.error(f"Error in vectorize_text_apply for '{text_col}': {e}")
        sys.exit(f"Critical error: {e}")


def sanitize_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """
    Applies a trained `CountVectorizer` to transform a specified text column into token frequencies.

    - Uses a pre-trained `CountVectorizer` to transform `df[text_col]` into token frequencies.
    - Retains only tokens specified in `keep_tokens` to ensure consistency with training data.
    - Automatically adds missing columns (if any) with zero values to maintain feature alignment.
    - Optionally applies a prefix to the tokenized column names for better identification.

    Parameters:
        df (pd.DataFrame): Input DataFrame containing the text column.
        text_col (str): Column name containing space-separated text tokens.
        vectorizer (CountVectorizer): Pre-trained `CountVectorizer` instance.
        keep_tokens (list): List of tokens to retain in the output.
        prefix (str, optional): Prefix to apply to column names (default: "").

    Returns:
        pd.DataFrame: Token frequency matrix with aligned columns.

    Raises:
        SystemExit: If an error occurs, logs the error and exits the program.
    """
    
    new_cols = []
    for c in df.columns:
        c_str = str(c)
        for bad in ['[', ']', '<', '>']:
            c_str = c_str.replace(bad, '_')
        new_cols.append(c_str)
    df.columns = new_cols
    return df


def load_model_assets(model_directory: str) -> dict:
    """
    Loads trained model assets from the specified directory.

    - Searches for a model file (`final_model.json` or `xgboost_model.json`).
    - Loads the corresponding trained XGBoost booster.
    - Retrieves the feature names list to maintain model input consistency.
    - Loads the best threshold for classification decisions, defaulting to 0.5 if unavailable.

    Parameters:
        model_directory (str): Path to the directory containing trained model files.

    Returns:
        dict: A dictionary containing:
            - 'booster' (xgb.Booster): Loaded XGBoost model.
            - 'feature_names' (list): List of expected feature names.
            - 'threshold' (float): Threshold for converting probabilities into binary labels.

    Raises:
        SystemExit: If model files or feature names cannot be found or loaded.
    """
    
    result = {}

    possible_model_files = ['final_model.json', 'xgboost_model.json']
    model_path = None
    for pmf in possible_model_files:
        path_check = os.path.join(model_directory, pmf)
        if os.path.exists(path_check):
            model_path = path_check
            break

    if not model_path:
        logging.error(f"No model file found in {model_directory}")
        sys.exit(f"Critical error: No model file found in {model_directory}")

    booster = xgb.Booster()
    try:
        booster.load_model(model_path)
        logging.info(f"Loaded XGBoost model from {model_path}")
        result['booster'] = booster
    except Exception as e:
        logging.error(f"Error loading model from {model_path}: {e}")
        sys.exit(f"Critical error: Error loading model: {e}")

    feature_names_path = os.path.join(model_directory, 'feature_names.joblib')
    if not os.path.exists(feature_names_path):
        logging.error(f"Feature names file not found in {model_directory}")
        sys.exit(f"Critical error: Feature names not found in {model_directory}")

    try:
        feature_names = joblib.load(feature_names_path)
        if not isinstance(feature_names, list):
            sys.exit(f"Critical error: feature_names in {feature_names_path} is not a list.")
        logging.info(f"Loaded {len(feature_names)} feature names from {feature_names_path}")
        result['feature_names'] = feature_names
    except Exception as e:
        logging.error(f"Error loading feature names: {e}")
        sys.exit(f"Critical error: {e}")

    threshold_file = os.path.join(model_directory, 'best_threshold.txt')
    if os.path.exists(threshold_file):
        try:
            with open(threshold_file, 'r') as f:
                th_str = f.read().strip()
            threshold_val = float(th_str)
            threshold_val = min(max(threshold_val, 0.0), 1.0)
            result['threshold'] = threshold_val
            logging.info(f"Loaded best threshold: {threshold_val} from {threshold_file}")
        except:
            logging.warning(f"Error reading threshold from {threshold_file}, defaulting to 0.5")
            result['threshold'] = 0.5
    else:
        result['threshold'] = 0.5
        logging.info("best_threshold.txt not found, defaulting to 0.5")

    return result


def map_predictions(pred_array: np.ndarray) -> np.ndarray:
    """
    Converts binary predictions (0/1) into human-readable BBB classification labels.

    - Maps `1` to `'BBB-'` (indicating poor BBB permeability).
    - Maps `0` to `'BBB+'` (indicating good BBB permeability).
    - Ensures correct label mapping, accounting for any potential inversion.

    Parameters:
        pred_array (np.ndarray): A NumPy array containing binary classification values (0 or 1).

    Returns:
        np.ndarray: A NumPy array of corresponding string labels (`'BBB+'` or `'BBB-'`).
    """

    mapped = np.where(pred_array == 1, 'BBB-', 'BBB+')
    return mapped


def main():
    """
    Main function for predicting BBB+/- classification using pre-trained XGBoost models.

    - Loads and preprocesses molecular data from a JSON file.
    - Applies trained vectorizers to extract fragment-based features (BRICs, RINGS, SIDE_CHAINS).
    - Handles missing values using KNN imputation.
    - Loads trained XGBoost models from a directory (supports multiple subfolders with different models).
    - Computes predictions and probabilities for each model.
    - Aggregates predictions using mean and median probability methods.
    - Saves results as a CSV file.

    Parameters:
        None (retrieves user input via command-line arguments or interactive prompts).

    Returns:
        None (outputs a CSV file containing predictions and logs process information).
    """
    
    parser = argparse.ArgumentParser(description="Predict BBB+/- Using Trained XGBoost Model(s)")
    parser.add_argument("--model_dir", help="Directory containing trained model assets (may have multiple subfolders).")
    parser.add_argument("--input_json", help="Path to the input JSON file for prediction.")
    parser.add_argument("--output_csv", help="Output CSV file to store predictions.")
    parser.add_argument("--use_knn", action='store_true', help="Enable KNN imputation on combined training and prediction data.")
    parser.add_argument("--knn_neighbors", type=int, default=DEFAULT_KNN_NEIGHBOURS, help="Number of neighbors for KNN imputation (default: 5).")

    args = parser.parse_args()

    
    if not args.model_dir:
        model_dir = input("Enter the directory containing trained model assets: ").strip()
    else:
        model_dir = args.model_dir

    if not args.input_json:
        input_json = input("Enter the path to the input JSON file for prediction: ").strip()
    else:
        input_json = args.input_json

    if not args.output_csv:
        output_csv = input("Enter the path to save the prediction CSV: ").strip()
    else:
        output_csv = args.output_csv
        
    if not args.use_knn:
        use_knn_input = input("Use KNN for missing data from Training and Prediction data? (y/n): ").strip().lower()
        use_knn = use_knn_input in ['y', 'yes']
    else:
        use_knn = args.use_knn


    os.makedirs(os.path.dirname(output_csv), exist_ok=True)

    
    log_file_path = os.path.join(os.path.dirname(output_csv), 'prediction.log')
    setup_logging(log_file_path)
    logging.info("Started prediction script")

    
    raw_data = load_data(input_json)
    df = preprocess_data(raw_data)
    df = handle_missing_bbb_column(df)
    df = clean_bbb_labels(df)

    if 'SMILES' not in df.columns:
        logging.error("No 'SMILES' column in data after preprocessing. Exiting.")
        sys.exit("Critical error: 'SMILES' column is missing.")

    
    vectorizer_brics_path = os.path.join(model_dir, "vectorizer_brics.joblib")
    keep_tokens_brics_path = os.path.join(model_dir, "kept_tokens_brics.joblib")

    vectorizer_rings_path = os.path.join(model_dir, "vectorizer_rings.joblib")
    keep_tokens_rings_path = os.path.join(model_dir, "kept_tokens_rings.joblib")

    vectorizer_side_path = os.path.join(model_dir, "vectorizer_sidechains.joblib")
    keep_tokens_side_path = os.path.join(model_dir, "kept_tokens_sidechains.joblib")

    for path_ in [
        vectorizer_brics_path, keep_tokens_brics_path,
        vectorizer_rings_path, keep_tokens_rings_path,
        vectorizer_side_path, keep_tokens_side_path
    ]:
        if not os.path.exists(path_):
            logging.error(f"Missing required vectorizer or tokens file: {path_}")
            sys.exit(f"Critical error: Missing {path_}")

    
    try:
        vectorizer_brics = joblib.load(vectorizer_brics_path)
        keep_tokens_brics = joblib.load(keep_tokens_brics_path)

        vectorizer_rings = joblib.load(vectorizer_rings_path)
        keep_tokens_rings = joblib.load(keep_tokens_rings_path)

        vectorizer_side = joblib.load(vectorizer_side_path)
        keep_tokens_side = joblib.load(keep_tokens_side_path)

        logging.info("Loaded vectorizers and kept_tokens for BRICs, RINGS, SIDE_CHAINS.")
    except Exception as e:
        logging.error(f"Error loading vectorizers or kept tokens: {e}")
        sys.exit(f"Critical error: {e}")

    
    drop_cols = ['NO.', 'BBB', 'SMILES', 'BRIC_SMILES', 'RINGS', 'SIDE_CHAINS']
    df_brics = vectorize_text_apply(df, 'BRIC_SMILES', vectorizer_brics, keep_tokens_brics, prefix="BRICS_")
    df_rings = vectorize_text_apply(df, 'RINGS_SMILES', vectorizer_rings, keep_tokens_rings, prefix="RINGS_")
    df_side  = vectorize_text_apply(df, 'SIDE_CHAINS_SMILES', vectorizer_side, keep_tokens_side, prefix="SIDECHAINS_")

    X_numeric = df.drop(columns=drop_cols, errors='ignore').copy()
    X_numeric = X_numeric.apply(pd.to_numeric, errors='coerce').fillna(0)

    X_full = pd.concat([X_numeric, df_brics, df_rings, df_side], axis=1)
    X_full = X_full.loc[:, ~X_full.columns.duplicated()].copy()


    if use_knn:
        # Path to the processed training features
        train_features_path = os.path.join(model_dir, 'processed_train_features.csv')
        if not os.path.exists(train_features_path):
            logging.error(f"Processed training features not found at {train_features_path}")
            sys.exit(f"Critical error: Processed training features not found at {train_features_path}")
        
        # Load processed training features
        X_train_processed = pd.read_csv(train_features_path)
        logging.info(f"Loaded processed training features from {train_features_path}, shape: {X_train_processed.shape}")
        
        # Combine training and prediction data for KNN imputation
        combined_X = pd.concat([X_train_processed, X_full], axis=0)
        logging.info(f"Combined training and prediction data for KNN imputation, shape: {combined_X.shape}")
        
        # Perform KNN imputation on combined data
        imputer = KNNImputer(n_neighbors=args.knn_neighbors)
        imputed_combined = imputer.fit_transform(combined_X)
        imputed_combined_df = pd.DataFrame(imputed_combined, columns=combined_X.columns)
        logging.info(f"KNN imputation complete, imputed data shape: {imputed_combined_df.shape}")
        
        # Separate back into training and prediction data
        imputed_train = imputed_combined_df.iloc[:X_train_processed.shape[0], :].copy()
        imputed_prediction = imputed_combined_df.iloc[X_train_processed.shape[0]:, :].copy()
        logging.info(f"Separated imputed data: training shape {imputed_train.shape}, prediction shape {imputed_prediction.shape}")
        
        # Replace X_full with imputed_prediction
        X_full = imputed_prediction
    else:
        # Existing KNN imputation on prediction data only
        numeric_cols = list(X_full.select_dtypes(include=[np.number]).columns)
        if len(numeric_cols) > 1 and X_full.shape[0] > 1:
            imputer = KNNImputer(n_neighbors=args.knn_neighbors)
            X_full[numeric_cols] = imputer.fit_transform(X_full[numeric_cols])
        else:
            X_full.fillna(0, inplace=True)

    X_full = sanitize_column_names(X_full)

    
    subfolders = [f.path for f in os.scandir(model_dir) if f.is_dir()]
    valid_subfolders = []
    for sf in subfolders:
        model_file = None
        for mf in ['final_model.json', 'xgboost_model.json']:
            if os.path.exists(os.path.join(sf, mf)):
                model_file = os.path.join(sf, mf)
                break
        feature_names_file = os.path.join(sf, 'feature_names.joblib')
        if model_file and os.path.exists(feature_names_file):
            valid_subfolders.append(sf)

    if not valid_subfolders:
        logging.info("No subfolders with valid models found; using single-model scenario in the main directory.")
        valid_subfolders = [model_dir]

    predictions_dict = {}
    probabilities_dict = {}
    metric_labels = []
    thresholds = []

    
    for folder in valid_subfolders:
        metric_name = os.path.basename(folder).lower()
        if folder == model_dir:
            metric_name = "default"

        logging.info(f"Loading model assets from: {folder}")
        assets = load_model_assets(folder)

        feature_names = assets['feature_names']
        threshold = assets['threshold']
        thresholds.append(threshold)

        missing_feats = [f for f in feature_names if f not in X_full.columns]
        extra_feats = [f for f in X_full.columns if f not in feature_names]

        if missing_feats:
            for mf in missing_feats:
                X_full[mf] = 0.0
        if extra_feats:
            X_model = X_full.drop(columns=extra_feats)
        else:
            X_model = X_full.copy()

        X_model = X_model[feature_names]

        dmatrix = xgb.DMatrix(X_model, feature_names=feature_names)
        y_probs = assets['booster'].predict(dmatrix)
        threshold = assets['threshold']

        if isinstance(y_probs, float):
            y_probs = np.full(shape=(X_model.shape[0],), fill_value=y_probs)
        elif isinstance(y_probs, list):
            y_probs = np.array(y_probs)
        elif isinstance(y_probs, np.ndarray):
            pass
        else:
            y_probs = np.array(y_probs)

        y_pred = (y_probs >= threshold).astype(int)
        metric_labels.append(metric_name)
        predictions_dict[metric_name] = y_pred
        probabilities_dict[metric_name] = y_probs

    
    output_df = pd.DataFrame()
    output_df['SMILES'] = df['SMILES']

    if 'NO.' in df.columns:
        output_df['NO.'] = df['NO.']

    if 'BBB' in df.columns:
        output_df['True_BBB'] = df['BBB']

    for metric_name in metric_labels:
        pred_col = f"Pred_{metric_name}"
        prob_col = f"Prob_{metric_name}"
        output_df[prob_col] = probabilities_dict[metric_name]
        mapped_pred = map_predictions(predictions_dict[metric_name])
        output_df[pred_col] = mapped_pred

    if metric_labels:
        prob_cols = [f"Prob_{mn}" for mn in metric_labels]
        average_threshold = np.mean(thresholds)
        output_df['Mean Probability'] = output_df[prob_cols].mean(axis=1)
        output_df['Mean BBB+/-'] = np.where(output_df['Mean Probability'] <= average_threshold, 'BBB+', 'BBB-')
        output_df['Median Probability'] = output_df[prob_cols].median(axis=1)
        output_df['Median BBB+/-'] = np.where(output_df['Median Probability'] <= average_threshold, 'BBB+', 'BBB-')
    else:
        output_df['Mean Probability'] = np.nan
        output_df['Mean BBB+/-'] = np.nan
        output_df['Median Probability'] = np.nan
        output_df['Median BBB+/-'] = np.nan

    try:
        output_df.to_csv(output_csv, index=False)
        logging.info(f"Predictions saved at: {output_csv}")
        print(f"Predictions have been saved to {output_csv}")
    except Exception as e:
        logging.error(f"Error saving predictions CSV: {e}")
        sys.exit(f"Critical error: {e}")

    logging.info("Prediction script completed successfully.")


if __name__ == "__main__":
    main()
