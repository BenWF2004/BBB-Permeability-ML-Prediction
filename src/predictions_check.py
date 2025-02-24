"""
Author: Ben Franey
Version: 1.1 - Publish: 1.0
Last Review Date: 30-01-2025
Overview:
Test script only.
To review predicions made where a True BBB colum exists.
"""

import pandas as pd
from sklearn.metrics import roc_auc_score, matthews_corrcoef, f1_score, confusion_matrix
import os

def calculate_metrics_from_csv(file_path):
    """
    Calculate AUC, MCC, F1, SN, SP, Mean Probability, and Median Probability for all models in a CSV file.

    Parameters:
        file_path (str): Path to the CSV file.

    Returns:
        dict: Metrics for each model.
    """
    # Load the data
    data = pd.read_csv(file_path)

    # Identify the true label column (contains 'True')
    true_label_col = [col for col in data.columns if 'True' in col]
    if not true_label_col:
        raise ValueError("No column containing 'True' found for true labels.")
    true_label_col = true_label_col[0]

    # Map true labels to binary (adjust mapping as per your dataset)
    label_mapping = {"BBB+": 0, "BBB-": 1}
    true_labels = data[true_label_col].map(label_mapping)
    if true_labels.isnull().any():
        raise ValueError("True labels contain values outside the mapping keys.")

    # Identify all models by finding unique suffixes after 'Prob_' and 'Pred_'
    prob_cols = [col for col in data.columns if col.startswith('Prob_')]
    pred_cols = [col for col in data.columns if col.startswith('Pred_')]

    # Extract model names by removing 'Prob_' and 'Pred_' prefixes
    models = set(col.replace('Prob_', '') for col in prob_cols).intersection(
             set(col.replace('Pred_', '') for col in pred_cols))
    
    if not models:
        raise ValueError("No matching 'Prob_' and 'Pred_' columns found for any model.")

    # Function to calculate individual metrics
    def calculate_individual_metrics(true, pred, pred_prob=None):
        cm = confusion_matrix(true, pred)
        if cm.shape == (2, 2):
            tn, fp, fn, tp = cm.ravel()
            sn = tp / (tp + fn) if (tp + fn) > 0 else 0
            sp = tn / (tn + fp) if (tn + fp) > 0 else 0
        else:
            # Handle cases where one of the classes might be missing
            tn = fp = fn = tp = 0
            sn = sp = 0
        auc = roc_auc_score(true, pred_prob) if pred_prob is not None else None
        mcc = matthews_corrcoef(true, pred) if len(set(true)) > 1 else None
        f1 = f1_score(true, pred) if len(set(true)) > 1 else None
        return sn, sp, auc, mcc, f1

    # Initialize results dictionary
    results = {}

    for model in sorted(models):
        #if "Mean" not in model and "Median" not in model:
        prob_col = f'Prob_{model}'
        pred_col = f'Pred_{model}'

        if prob_col not in data.columns or pred_col not in data.columns:
            print(f"Skipping model '{model}' due to missing columns.")
            continue

        # Extract probability and prediction columns
        prob_predictions = pd.to_numeric(data[prob_col], errors='coerce')
        if prob_predictions.isnull().any():
            raise ValueError(f"Probabilities for model '{model}' contain non-numeric values.")

        # Convert probabilities to binary predictions using a 0.5 threshold
        binary_predictions = (prob_predictions > 0.5).astype(int)

        # Map predicted labels to binary
        pred_labels = data[pred_col].map(label_mapping)
        if pred_labels.isnull().any():
            raise ValueError(f"Predictions for model '{model}' contain values outside the mapping keys.")

        # Calculate metrics
        sn, sp, auc, mcc, f1 = calculate_individual_metrics(true_labels, binary_predictions, prob_predictions)
        
        # Calculate mean and median probabilities
        mean_prob = prob_predictions.mean()
        median_prob = prob_predictions.median()

        # Store metrics
        results[model] = {
            "SN": round(sn, 3),
            "SP": round(sp, 3),
            "AUC": round(auc, 3) if auc is not None else None,
            "MCC": round(mcc, 3) if mcc is not None else None,
            "F1": round(f1, 3) if f1 is not None else None,
        }

    return results

def generate_comparative_table(files):
    """
    Generate a comparative table of metrics for multiple CSV files.

    Parameters:
        files (list): List of CSV file names.

    Returns:
        pd.DataFrame: Comparative metrics table.
    """
    comparison_table = []

    for file_name in files:
        if os.path.exists(file_name):
            try:
                metrics = calculate_metrics_from_csv(file_name)
                for model, metric_values in metrics.items():
                    row = {"File": file_name, "Model": model}
                    row.update(metric_values)
                    comparison_table.append(row)
            except Exception as e:
                print(f"Error processing {file_name}: {e}")
        else:
            print(f"File not found: {file_name}")

    # Convert to DataFrame for a tabular format
    comparison_df = pd.DataFrame(comparison_table)
    return comparison_df

if __name__ == "__main__":
    # Ask the user to input CSV filenames, separated by commas
    file_input = input("Enter CSV filenames (comma-separated): ").strip()
    file_output = input("Enter CSV filename for saved metrics (default: predictions/comparative_metrics.csv): ").strip()
    
    #file_input = "avg_none.csv,avg_smote.csv,avg_smoteenn.csv,avg_smotetomek.csv,both_none.csv,both_smote.csv,both_smoteenn.csv,both_smotetomek.csv,knn_avg_none.csv,knn_avg_smote.csv,knn_avg_smoteenn.csv,knn_avg_smotetomek.csv,knn_both_none.csv,knn_both_smote.csv,knn_pubchem-else-rdkit_none.csv,knn_pubchem-else-rdkit_smote.csv,knn_pubchem-else-rdkit_smoteenn.csv,knn_pubchem-else-rdkit_smotetomek.csv,knn_pubchem-only_none.csv,knn_pubchem-only_smote.csv,knn_pubchem-only_smoteenn.csv,knn_pubchem-only_smotetomek.csv,knn_rdkit-only_none.csv,knn_rdkit-only_smote.csv,knn_rdkit-only_smoteenn.csv,knn_rdkit-only_smotetomek.csv,pubchem-else-rdkit_none.csv,pubchem-else-rdkit_smote.csv,pubchem-else-rdkit_smoteenn.csv,pubchem-else-rdkit_smotetomek.csv,pubchem-only_none.csv,pubchem-only_smote.csv,pubchem-only_smoteenn.csv,pubchem-only_smotetomek.csv,rdkit-only-pred_avg_none.csv,rdkit-only-pred_avg_smote.csv,rdkit-only-pred_avg_smoteenn.csv,rdkit-only-pred_avg_smotetomek.csv,rdkit-only-pred_both_none.csv,rdkit-only-pred_both_smote.csv,rdkit-only-pred_both_smoteenn.csv,rdkit-only-pred_both_smotetomek.csv,rdkit-only-pred_knn_avg_none.csv,rdkit-only-pred_knn_avg_smote.csv,rdkit-only-pred_knn_avg_smoteenn.csv,rdkit-only-pred_knn_avg_smotetomek.csv,rdkit-only-pred_knn_both_none.csv,rdkit-only-pred_knn_both_smote.csv,rdkit-only-pred_knn_both_smoteenn.csv,rdkit-only-pred_knn_both_smotetomek.csv,rdkit-only-pred_knn_pubchem-else-rdkit_none.csv,rdkit-only-pred_knn_pubchem-else-rdkit_smote.csv,rdkit-only-pred_knn_pubchem-else-rdkit_smoteenn.csv,rdkit-only-pred_knn_pubchem-else-rdkit_smotetomek.csv,rdkit-only-pred_knn_pubchem-only_none.csv,rdkit-only-pred_knn_pubchem-only_smote.csv,rdkit-only-pred_knn_pubchem-only_smoteenn.csv,rdkit-only-pred_knn_pubchem-only_smotetomek.csv,rdkit-only-pred_knn_rdkit-only_none.csv,rdkit-only-pred_knn_rdkit-only_smote.csv,rdkit-only-pred_knn_rdkit-only_smoteenn.csv,rdkit-only-pred_knn_rdkit-only_smotetomek.csv,rdkit-only-pred_pubchem-else-rdkit_none.csv,rdkit-only-pred_pubchem-else-rdkit_smote.csv,rdkit-only-pred_pubchem-else-rdkit_smoteenn.csv,rdkit-only-pred_pubchem-else-rdkit_smotetomek.csv,rdkit-only-pred_pubchem-only_none.csv,rdkit-only-pred_pubchem-only_smote.csv,rdkit-only-pred_pubchem-only_smoteenn.csv,rdkit-only-pred_pubchem-only_smotetomek.csv,rdkit-only-pred_rdkit-only_none.csv,rdkit-only-pred_rdkit-only_smote.csv,rdkit-only-pred_rdkit-only_smoteenn.csv,rdkit-only-pred_rdkit-only_smotetomek.csv,rdkit-only_none.csv,rdkit-only_smote.csv,rdkit-only_smoteenn.csv,rdkit-only_smotetomek.csv"
    #file_output = "predictions/model_evaluations.csv"
    
    if not file_output:
        file_output = "predictions/comparative_metrics.csv"

    # Convert input string to a list, removing any whitespace
    files = [file.strip() for file in file_input.split(",")]

    #files = ["predictions/evaluation-single/" + file.strip() for file in file_input.split(",")]

    # Generate the comparative table
    comparison_df = generate_comparative_table(files)

    # Display the table
    print(comparison_df)

    # Save the table to a CSV file
    comparison_df.to_csv(file_output, index=False)
    print(f"\nComparative metrics saved to {file_output}")
