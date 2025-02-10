"""
Author: Ben Franey
Version: 11.2.0 - Publish: 1.1
Last Review Date: 06-02-2025
Overview:
    Predict BBB+/- labels for molecules using pre-trained XGBoost models.
    Adjusts mean and median probability calculations based on individual model thresholds.
    Outputs all model prediction probabilities and labels alongside aggregated binary labels.
    
Usage example:  
python3 src/predict.py \
  --model_path o/output1/best_model.pth \
  --input_json data/example_prediction/model_ready.json \
  --output_csv results/predictions.csv \
  --use_knn y
  
Available Arguments (argparse)
--model_path: Path to the trained model file.
--input_json: Path to the preprocessed input data in JSON format.
--output_csv: Path to save the predictions.
--use_knn: y/n for use of KNN.
--knn_neighbors: Number of neighbors for KNN imputation (default: 5).
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
            # If not a list, check for dictionary with 'molecules'
            if isinstance(data, dict) and 'molecules' in data and isinstance(data['molecules'], list):
                data = data['molecules']
            else:
                sys.exit("Critical error: JSON structure not recognised. "
                         "Expecting a list or { 'molecules': [...] }.")
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
            # Skip if no SMILES present
            continue

        row_data = {
            'NO.': entry.get('NO.', np.nan),
            'BBB': entry.get('BBB+/BBB-', np.nan),
            'SMILES': entry.get('SMILES', np.nan),

            # Use PubChem is valid, else RDKit
            'LogP': entry.get('LogP_PubChem', np.nan)
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
                else entry.get('AtomStereo_RDKit', np.nan),

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


def vectorize_text_apply(
    df: pd.DataFrame,
    text_col: str,
    vectorizer: CountVectorizer,
    keep_tokens: list,
    prefix: str = ""
) -> pd.DataFrame:
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

        # Add missing columns
        missing = [tk for tk in keep_tokens if tk not in temp_df.columns]
        for mc in missing:
            temp_df[mc] = 0

        # Retain only kept tokens
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
    return np.where(pred_array == 0, 'BBB+', 'BBB-')


def load_subfolder_assets(folder_path: str) -> dict:
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
    # Check for model file
    possible_model_files = ['final_model.json', 'xgboost_model.json']
    model_path = None
    for pmf in possible_model_files:
        check_path = os.path.join(folder_path, pmf)
        if os.path.exists(check_path):
            model_path = check_path
            break
    
    if not model_path:
        return None  # Not a valid subfolder

    # Check for feature_names
    feature_names_path = os.path.join(folder_path, 'feature_names.joblib')
    if not os.path.exists(feature_names_path):
        return None

    # Check for vectorizers & tokens
    vb = os.path.join(folder_path, 'vectorizer_brics.joblib')
    kb = os.path.join(folder_path, 'kept_tokens_brics.joblib')
    vr = os.path.join(folder_path, 'vectorizer_rings.joblib')
    kr = os.path.join(folder_path, 'kept_tokens_rings.joblib')
    vs = os.path.join(folder_path, 'vectorizer_sidechains.joblib')
    ks = os.path.join(folder_path, 'kept_tokens_sidechains.joblib')
    vectorizer_files = [vb, kb, vr, kr, vs, ks]
    for vf in vectorizer_files:
        if not os.path.exists(vf):
            return None

    # Attempt to load everything
    try:
        booster = xgboost_load_model(model_path)
        feature_names = joblib.load(feature_names_path)

        vect_brics = joblib.load(vb)
        keep_brics = joblib.load(kb)
        vect_rings = joblib.load(vr)
        keep_rings = joblib.load(kr)
        vect_side  = joblib.load(vs)
        keep_side  = joblib.load(ks)

        # Load threshold
        threshold_file = os.path.join(folder_path, 'best_threshold.txt')
        if os.path.exists(threshold_file):
            with open(threshold_file, 'r') as f:
                th_str = f.read().strip()
            try:
                threshold_val = float(th_str)
                threshold_val = min(max(threshold_val, 0.0), 1.0)
            except:
                threshold_val = 0.5
        else:
            threshold_val = 0.5

        result = {
            'model_path': model_path,
            'booster': booster,
            'feature_names': feature_names,
            'vect_brics': vect_brics,
            'keep_brics': keep_brics,
            'vect_rings': vect_rings,
            'keep_rings': keep_rings,
            'vect_side': vect_side,
            'keep_side': keep_side,
            'threshold': threshold_val
        }
        return result
    except Exception as e:
        logging.error(f"Error loading assets in {folder_path}: {e}")
        return None


def xgboost_load_model(model_path: str) -> xgb.Booster:
    """
    Loads an XGBoost model from a JSON file.

    - Instantiates an empty XGBoost Booster object.
    - Loads the model parameters and structure from the specified JSON file.
    - Logs a confirmation message indicating successful model loading.

    Parameters:
        model_path (str): Path to the JSON file containing the XGBoost model.

    Returns:
        xgb.Booster: The loaded XGBoost model.
    """
    
    booster = xgb.Booster()
    booster.load_model(model_path)
    logging.info(f"Loaded XGBoost model from {model_path}")
    return booster


def main():
    """
    Main function for predicting BBB+/- classification using pre-trained XGBoost models.
    Handles multiple subfolders (one model per subfolder) or a single-model scenario.
    Aggregates all model predictions by mean/median probability.
    """
    
    parser = argparse.ArgumentParser(description="Predict BBB+/- Using Trained XGBoost Model(s)")
    parser.add_argument("--model_dir", help="Directory containing trained model assets (one or more subfolders).")
    parser.add_argument("--input_json", help="Path to the input JSON file for prediction.")
    parser.add_argument("--output_csv", help="Output CSV file to store predictions.")
    parser.add_argument("--use_knn", type=str, choices=['y', 'n'],
                    help="Enable ('y') or disable ('n') KNN imputation.")
    parser.add_argument("--knn_neighbors", type=int, default=DEFAULT_KNN_NEIGHBOURS,
                        help="Number of neighbors for KNN imputation (default: 5).")

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
        use_knn = use_knn_input in ['y']
    else:
        use_knn = args.use_knn

    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    log_file_path = os.path.join(os.path.dirname(output_csv), 'prediction.log')
    setup_logging(log_file_path)
    logging.info("Started prediction script")

    # 1. Load and preprocess data
    raw_data = load_data(input_json)
    df = preprocess_data(raw_data)
    df = handle_missing_bbb_column(df)
    df = clean_bbb_labels(df)

    if 'SMILES' not in df.columns:
        logging.error("No 'SMILES' column in data after preprocessing. Exiting.")
        sys.exit("Critical error: 'SMILES' column is missing.")

    # 2. Find valid subfolders - A valid subfolder has model + feature_names + vectorizers + tokens
    subfolders = [
        f.path for f in os.scandir(model_dir)
        if f.is_dir() and not f.name.startswith('.')
    ]
    valid_subfolder_assets = []
    for sf in subfolders:
        assets = load_subfolder_assets(sf)
        if assets is not None:
            valid_subfolder_assets.append((sf, assets))

    # 3. If no valid subfolders, check if the top-level is a single-model scenario
    top_level_assets = None
    if not valid_subfolder_assets:
        logging.info("No valid subfolders found. Checking single-model scenario at top level...")
        assets = load_subfolder_assets(model_dir)
        if assets is not None:
            top_level_assets = assets

    if not valid_subfolder_assets and not top_level_assets:
        logging.error(f"No valid model(s) found in '{model_dir}'")
        sys.exit(f"Critical error: No valid model(s) found in '{model_dir}'")

    # Store (model_label -> results) for aggregator
    predictions_dict = {}
    probabilities_dict = {}
    thresholds_used = []
    model_labels = []

    # 4. Function to build (and optionally KNN-impute) the feature matrix for a given subfolder’s vectorizers
    def build_feature_matrix_for_folder(assets_dict) -> pd.DataFrame:
        """
        Applies the folder’s vectorizers/tokens to the raw DataFrame, returns X_full
        possibly after combined KNN with train_matrix.csv in that folder.
        """
        # Vectorizers
        vect_brics = assets_dict['vect_brics']
        keep_brics = assets_dict['keep_brics']
        vect_rings = assets_dict['vect_rings']
        keep_rings = assets_dict['keep_rings']
        vect_side  = assets_dict['vect_side']
        keep_side  = assets_dict['keep_side']

        # Build fragment data
        df_brics = vectorize_text_apply(df, 'BRIC_SMILES', vect_brics, keep_brics, prefix="BRICS_")
        df_rings = vectorize_text_apply(df, 'RINGS_SMILES', vect_rings, keep_rings, prefix="RINGS_")
        df_side  = vectorize_text_apply(df, 'SIDE_CHAINS_SMILES', vect_side, keep_side, prefix="SIDECHAINS_")

        drop_cols = [
            'NO.', 'BBB', 'SMILES', 'LogBB', 'Group', 'BBB_Label',
            'BRIC_SMILES', 'RINGS_SMILES', 'SIDE_CHAINS_SMILES'
        ]
        X_numeric = df.drop(columns=drop_cols, errors='ignore').copy()
        X_numeric = X_numeric.apply(pd.to_numeric, errors='coerce').fillna(0)

        X_concat = pd.concat([X_numeric, df_brics, df_rings, df_side], axis=1)
        X_concat = X_concat.loc[:, ~X_concat.columns.duplicated()].copy()

        # Combined KNN if requested
        if use_knn == 'y': 
            train_matrix_path = os.path.join(os.path.dirname(assets_dict['model_path']), 'train_matrix.csv')
            if not os.path.exists(train_matrix_path):
                logging.error(f"train_matrix.csv not found at {train_matrix_path}, cannot do combined KNN.")
                sys.exit(f"Critical error: Missing {train_matrix_path} for combined KNN.")
            try:
                X_train_processed = pd.read_csv(train_matrix_path)
                logging.info(f"Loaded training features for KNN from {train_matrix_path}, shape: {X_train_processed.shape}")

                combined_X = pd.concat([X_train_processed, X_concat], axis=0)
                logging.info(f"Combined train+predict data shape for KNN: {combined_X.shape}")

                imputer = KNNImputer(n_neighbors=args.knn_neighbors)
                imputed = imputer.fit_transform(combined_X)
                imputed_df = pd.DataFrame(imputed, columns=combined_X.columns)

                # Slice back out the portion that corresponds to X_concat
                X_concat_imputed = imputed_df.iloc[X_train_processed.shape[0]:, :].copy()
                logging.info(f"KNN-imputed new data shape: {X_concat_imputed.shape}")

                return sanitize_column_names(X_concat_imputed)
            except Exception as e:
                logging.error(f"Error during combined KNN with {train_matrix_path}: {e}")
                sys.exit(f"Critical error: {e}")
        else:
            # KNN just on the new data alone (optional):
            numeric_cols = list(X_concat.select_dtypes(include=[np.number]).columns)
            if len(numeric_cols) > 1 and X_concat.shape[0] > 1:
                imputer = KNNImputer(n_neighbors=args.knn_neighbors)
                X_concat[numeric_cols] = imputer.fit_transform(X_concat[numeric_cols])
            else:
                X_concat.fillna(0, inplace=True)
            return sanitize_column_names(X_concat)

    # 5. Helper to run predictions for a single subfolder
    def predict_with_subfolder(assets_dict, X_master: pd.DataFrame, label: str):
        booster = assets_dict['booster']
        feature_names = assets_dict['feature_names']
        threshold = assets_dict['threshold']

        # Align columns
        missing_feats = [f for f in feature_names if f not in X_master.columns]
        for mf in missing_feats:
            X_master[mf] = 0.0
        extra_feats = [f for f in X_master.columns if f not in feature_names]
        if extra_feats:
            X_model = X_master.drop(columns=extra_feats)
        else:
            X_model = X_master.copy()

        # Reorder to exactly match training
        X_model = X_model[feature_names]
        dmatrix = xgb.DMatrix(X_model, feature_names=feature_names)
        y_probs = booster.predict(dmatrix)

        # Convert to array
        if isinstance(y_probs, float):
            y_probs = np.full(shape=(X_model.shape[0],), fill_value=y_probs)
        elif isinstance(y_probs, list):
            y_probs = np.array(y_probs)
        elif not isinstance(y_probs, np.ndarray):
            y_probs = np.array(y_probs)

        y_pred = (y_probs >= threshold).astype(int)

        model_labels.append(label)
        predictions_dict[label] = y_pred
        probabilities_dict[label] = y_probs
        thresholds_used.append(threshold)

    # 6. If we have multiple valid subfolders, run each
    if valid_subfolder_assets:
        for (sf_path, sf_assets) in valid_subfolder_assets:
            metric_name = os.path.basename(sf_path).lower()
            if not metric_name:
                # In case subfolder name is blank
                metric_name = "model_subfolder"

            logging.info(f"Building features for subfolder: {sf_path}")
            X_subfolder = build_feature_matrix_for_folder(sf_assets)
            predict_with_subfolder(sf_assets, X_subfolder, metric_name)

    # 7. If we have a top-level single-model
    if top_level_assets:
        label = "default"
        logging.info(f"Building features for single-model scenario: {model_dir}")
        X_single = build_feature_matrix_for_folder(top_level_assets)
        predict_with_subfolder(top_level_assets, X_single, label)

    # 8. Aggregate results
    output_df = pd.DataFrame()
    output_df['SMILES'] = df['SMILES']

    if 'NO.' in df.columns:
        output_df['NO.'] = df['NO.']

    if 'BBB' in df.columns:
        output_df['True_BBB'] = df['BBB']

    # For each model's predictions
    for m_label in model_labels:
        prob_col = f"Prob_{m_label}"
        pred_col = f"Pred_{m_label}"
        output_df[prob_col] = probabilities_dict[m_label]
        output_df[pred_col] = map_predictions(predictions_dict[m_label])

    # If multiple models, compute aggregator columns
    if model_labels:
        prob_cols = [f"Prob_{lbl}" for lbl in model_labels]
        # Average threshold across all loaded models
        avg_threshold = np.mean(thresholds_used)

        # Mean Probability approach
        output_df['Prob_Mean'] = output_df[prob_cols].mean(axis=1)
        # Classify as 'BBB+' if the mean probability is below or equal to the threshold, else 'BBB-'
        output_df['Pred_Mean'] = np.where(
            output_df['Prob_Mean'] <= avg_threshold, 'BBB+', 'BBB-'
        )

        # Median Probability approach
        output_df['Prob_Median'] = output_df[prob_cols].median(axis=1)
        # Classify as 'BBB+' if the mean probability is below or equal to the threshold, else 'BBB-'
        output_df['Pred_Median'] = np.where(
            output_df['Prob_Median'] <= avg_threshold, 'BBB+', 'BBB-'
        )
    else:
        # Should not happen unless no valid models
        output_df['Prob_Mean'] = np.nan
        output_df['Pred_Mean'] = np.nan
        output_df['Prob_Median'] = np.nan
        output_df['Pred_Median'] = np.nan

    # 9. Save results
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
