"""
Author: Ben Franey
Version: 5.0.1 - Publish: 1.0
Last Review Date: 29-01-2025
Overview:
Analyzes molecular data from CSV files, classifying fragments and comparing 
RDKit and PubChem descriptors. Generates statistical reports and visualizations.

Key Features:
- Data Processing & Classification:
  - Reads CSV files for molecules, BRICS, RINGS, and SIDECHAINS.
  - Filters and classifies fragments (BBB+ / BBB-) based on frequency and ratio.
  - Supports automated directory detection.

- Descriptor Analysis & Visualization:
  - Computes and compares RDKit and PubChem descriptors.
  - Generates histograms, boxplots, and scatter plots for molecular properties.
  - Performs RDKit vs. PubChem regression analysis with statistical metrics.

- Automated Output Generation:
  - Saves processed data as CSV files with classified fragments.
  - Creates structured summary reports in 'grouping' and 'comparisons' directories.
  - Logs missing data and descriptor inconsistencies.

Usage:
    python src/analyze.py \
        --parent_dir path/to/processed_data
"""

import os
import numpy as np
import pandas as pd
import argparse
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.stats import linregress, norm
import matplotlib.colors as mcolors
import matplotlib.cm as cm

# Color mapping for plot elements
GRAPH_COLOR_MAPPING = {
    'BBB+': 'teal',
    'BBB-': 'orange',
    'pair-wise-reflect': 'red',
    'pair-wise-data': 'navy',
    'grid-lines': 'lightgrey',
}

# Mapping of descriptor keys to their corresponding CSV column prefixes
DESCRIPTOR_MAPPING = {
    'mw':              'MW',
    'logp':            'LogP',
    'tpsa':            'TPSA',
    'hbond_acceptors': 'HBA',
    'hbond_donors':    'HBD',
    'flexibility':     'Flexibility',
    'charge':          'Charge',
    'atom_stereo':     'AtomStereo',
    'heavy_atoms':             'HeavyAtom',
    'atom_count_c':            'AtomCount_C',
    'atom_count_h':            'AtomCount_H',
    'atom_count_o':            'AtomCount_O',
    'atom_count_n':            'AtomCount_N',
    'atom_count_s':            'AtomCount_S',
    'atom_count_f':            'AtomCount_F',
    'atom_count_cl':           'AtomCount_Cl',
    'atom_count_br':           'AtomCount_Br',
    'atom_count_i':            'AtomCount_I',
    'atom_count_p':            'AtomCount_P',
    'atom_count_b':            'AtomCount_B',
    'atom_count_li':           'AtomCount_Li',
    'bond_count_single':       'BondCount_Single',
    'bond_count_double':       'BondCount_Double',
    'bond_count_triple':       'BondCount_Triple',
    'bond_count_aromatic':     'BondCount_Aromatic',
    'total_chiral_centers':    'Total_Chiral_Centers',
    'r_isomers':               'R_Isomers',
    's_isomers':               'S_Isomers',
    'e_isomers':               'E_Isomers',
    'z_isomers':               'Z_Isomers',
    'num_4_rings_aromatic':    'Num_4_Rings_Aromatic',
    'num_4_rings_nonaromatic': 'Num_4_Rings_NonAromatic',
    'num_4_rings_total':       'Num_4_Rings_Total',
    'num_5_rings_aromatic':    'Num_5_Rings_Aromatic',
    'num_5_rings_nonaromatic': 'Num_5_Rings_NonAromatic',
    'num_5_rings_total':       'Num_5_Rings_Total',
    'num_6_rings_aromatic':    'Num_6_Rings_Aromatic',
    'num_6_rings_nonaromatic': 'Num_6_Rings_NonAromatic',
    'num_6_rings_total':       'Num_6_Rings_Total',
    'num_8_rings_aromatic':    'Num_8_Rings_Aromatic',
    'num_8_rings_nonaromatic': 'Num_8_Rings_NonAromatic',
    'num_8_rings_total':       'Num_8_Rings_Total',
    'num_aromatic_rings':      'Num_Aromatic_Rings',
    'num_nonaromatic_rings':   'Num_NonAromatic_Rings',
    'num_total_rings':         'Num_Total_Rings',
    'wiener_index':            'WienerIndex',
    'eccentric_connectivity':  'EccentricConnectivityIndex',
    'radius_of_gyration':      'RadiusOfGyration',
}

VALID_DESCRIPTORS = list(DESCRIPTOR_MAPPING.keys())


def get_source_from_column(column_name: str) -> str:
    """
    Extracts the data source (RDKit or PubChem) from a column name.

    - Identifies whether the column corresponds to RDKit or PubChem descriptors.
    - Returns 'Unknown' if the source cannot be determined.

    Parameters:
        column_name (str): The name of the column.

    Returns:
        str: 'RDKit', 'PubChem', or 'Unknown'.
    """
    if '_RDKit' in column_name:
        return 'RDKit'
    elif '_PubChem' in column_name:
        return 'PubChem'
    return 'Unknown'

def plot_histograms(descriptor_key: str, df: pd.DataFrame, bbb_column: str, grouping_dir: str):
    """
    Generates histograms and boxplots for a given molecular descriptor 
    and saves them to the specified directory.

    - Uses RDKit and PubChem descriptors.
    - Categorizes data into BBB+ and BBB- groups.
    - Plots normal distributions over histograms if variance exists.
    - Saves output images with automated filename generation.

    Parameters:
        descriptor_key (str): Descriptor key from `DESCRIPTOR_MAPPING`.
        df (pd.DataFrame): DataFrame containing molecular data.
        bbb_column (str): Column name for BBB+ or BBB- classification.
        grouping_dir (str): Directory where images will be saved.

    Returns:
        None: Saves plots as files.
    """
    
    desc_prefix = DESCRIPTOR_MAPPING[descriptor_key]
    sources = ['RDKit', 'PubChem']

    for source in sources:
        # Construct column name for descriptor
        col = f"{desc_prefix}_{source}"
        
        # Skip if descriptor or BBB column is missing
        if col not in df.columns or bbb_column not in df.columns:
            continue
        
        # Extract relevant data, dropping NaN values
        data = df[[col, bbb_column]].dropna()
        if data.empty:
            continue

        # Separate data by BBB classification
        data_bbb_neg = data[data[bbb_column] == 'BBB-'][col]
        data_bbb_pos = data[data[bbb_column] == 'BBB+'][col]

        # Determine min/max values and range
        min_val = data[col].min()
        max_val = data[col].max()
        range_ = max_val - min_val

        # Determine bin width based on descriptor type
        if range_ <= 1e-9:
            bins = 'auto'
        else:
            if descriptor_key in [
                'hbond_acceptors', 'hbond_donors', 'flexibility', 'total_chiral_centers', 'charge'
            ] or descriptor_key.startswith(('atom_stereo', 'heavy_atoms', 'atom_count_', 'bond_count_')) \
               or descriptor_key.endswith('_isomers') or descriptor_key.startswith('num_'):
                bins = np.arange(min_val - 0.5, max_val + 1.5, 1)
            elif descriptor_key == 'tpsa':
                bins = np.arange(min_val - 0.5, max_val + 1.5, 12)
            elif descriptor_key == 'radius_of_gyration':
                bins = np.arange(min_val - 0.5, max_val + 1.5, 0.25)
            elif descriptor_key == 'eccentric_connectivity':
                bins = np.arange(min_val - 0.5, max_val + 1.5, (max_val - min_val + 2) / 100)
            elif descriptor_key == 'wiener_index':
                bins = np.arange(min_val - 0.5, max_val + 1.5, (max_val - min_val + 2) / 250)
            elif descriptor_key == 'mw':
                bins = np.arange(min_val - 0.5, max_val + 1.5, 10)
            elif np.issubdtype(data[col].dtype, np.number):
                bins = np.arange(min_val - 0.5, max_val + 1.5, 0.25)
            else:
                bins = 'auto'

        # Create subplot layout: histogram (top), boxplot (bottom)
        fig, (ax, ax_box) = plt.subplots(
            nrows=2,
            gridspec_kw={'height_ratios': [3, 1]},
            figsize=(18, 16),
            constrained_layout=True
        )

        # Plot histograms for BBB+ and BBB- groups
        ax.hist(
            data_bbb_neg,
            bins=bins,
            color=GRAPH_COLOR_MAPPING['BBB-'],
            label='BBB-',
            alpha=0.5,
            density=True
        )
        ax.hist(
            data_bbb_pos,
            bins=bins,
            color=GRAPH_COLOR_MAPPING['BBB+'],
            label='BBB+',
            alpha=0.5,
            density=True
        )
        
        # Ensures that the maximum density is 1
        ax.set_ylim(0, min(1, ax.get_ylim()[1]))

        # Overlay normal distributions
        for class_data, color, label_str in [
            (data_bbb_neg, GRAPH_COLOR_MAPPING['BBB-'], 'BBB-'),
            (data_bbb_pos, GRAPH_COLOR_MAPPING['BBB+'], 'BBB+')
        ]:
            if len(class_data) > 1:
                mean_ = class_data.mean()
                std_  = class_data.std()
                if std_ > 1e-9:
                    x_   = np.linspace(class_data.min(), class_data.max(), 1000)
                    pdf_ = norm.pdf(x_, mean_, std_)
                    ax.plot(
                        x_, pdf_, color=color, linestyle='solid',
                        label=f"{label_str} Normal Distribution", linewidth=4
                    )

        # Display statistical information on the plot
        combined_text = (
            f"BBB-\n"
            f"N: {len(data_bbb_neg)}\n"
            f"Mean: {data_bbb_neg.mean():.2f}\n"
            f"SD: {data_bbb_neg.std():.2f}\n\n"
            f"BBB+\n"
            f"N: {len(data_bbb_pos)}\n"
            f"Mean: {data_bbb_pos.mean():.2f}\n"
            f"SD: {data_bbb_pos.std():.2f}"
        )
        ax.text(
            0.98, 0.68,
            combined_text,
            transform=ax.transAxes, fontsize=18, color='black',
            verticalalignment='top', horizontalalignment='right',
            bbox=dict(facecolor='white', alpha=0.8)
        )

        # Configure main plot
        ax.grid(True, color=GRAPH_COLOR_MAPPING['grid-lines'], linestyle='--', linewidth=0.5)
        ax.set_ylabel('Density', fontsize=24)
        ax.set_xlabel(f'{desc_prefix} ({source})', fontsize=24)
        ax.tick_params(axis='both', which='major', labelsize=18)
        ax_box.tick_params(axis='both', which='major', labelsize=18)
        ax.legend(loc="upper right", fontsize=24, title="Legend and Metrics", title_fontsize=27)
        ax.set_title(f'Distribution of {desc_prefix} from {source} by BBB Classification', fontsize=30)

        # Create boxplot representation
        box_data   = [data_bbb_neg, data_bbb_pos]
        box_colors = [GRAPH_COLOR_MAPPING['BBB-'], GRAPH_COLOR_MAPPING['BBB+']]

        for i, (class_data, color) in enumerate(zip(box_data, box_colors), start=1):
            mean_ = class_data.mean() if len(class_data) > 0 else 0
            std_  = class_data.std()  if len(class_data) > 1 else 0
            ci_95 = 1.96 * std_

            # Add shaded confidence interval region
            ax_box.fill_betweenx([i - 0.2, i + 0.2], mean_ - std_, mean_ + std_, color=color, alpha=0.5)
            ax_box.plot([mean_ - ci_95, mean_ + ci_95], [i, i], color='black', linestyle='-', linewidth=1.5)
            
            # Custom defintion for max-min height and mean plotting
            ymin_vales = [0, 0.05, 0.7]
            ymax_vales = [0, 0.3, 0.95]
            ax_box.axvline(x=mean_, ymin=ymin_vales[i], ymax=ymax_vales[i], color='black', linewidth=2)

            # Highlight outliers
            outliers = class_data[(class_data < mean_ - ci_95) | (class_data > mean_ + ci_95)]
            ax_box.scatter(outliers, [i]*len(outliers), color='gray', alpha=0.6, marker='o')

        # Configure boxplot
        ax_box.set_yticks([1, 2])
        ax_box.set_yticklabels(['BBB-', 'BBB+'], fontsize=27)
        ax_box.set_xlabel(f'{desc_prefix} ({source})', fontsize=27)
        ax_box.set_title(f'Boxplot Overview of {desc_prefix} from {source}', fontsize=30)
        ax_box.grid(False)

        # Save figure to the specified directory
        histogram_filename = f"{descriptor_key}_{source}_histogram.png"
        histogram_filepath = os.path.join(grouping_dir, histogram_filename)
        os.makedirs(grouping_dir, exist_ok=True)
        plt.savefig(histogram_filepath)
        plt.close()
        print(f"Histogram saved => {histogram_filepath}")

def compare_and_plot_rdkit_vs_pubchem(
    descriptor_key: str,
    df: pd.DataFrame,
    comparisons_dir: str,
    function_column: str = "MW_RDKit"  # Renamed and defaulted to "MW_RDKit"
):
    """
    Generates scatter plots comparing RDKit and PubChem descriptor values, 
    with color mapping based on a specified function column.

    - Validates descriptor key, function column, and descriptor columns.
    - Computes differences, MAE, RMSE, and performs linear regression.
    - Uses a scatter plot to visualize RDKit vs. PubChem values.
    - Adds a reference y = x line and statistical metrics.
    - Saves both the plot and an analysis info file.

    Parameters:
        descriptor_key (str): Descriptor key from `DESCRIPTOR_MAPPING`.
        df (pd.DataFrame): DataFrame containing molecular descriptors.
        comparisons_dir (str): Directory where plots will be saved.
        function_column (str): Column name for color mapping (default: "MW_RDKit").

    Returns:
        None: Saves scatter plot and analysis info file.
    """
    # Validate descriptor
    descriptor_key = descriptor_key.lower()
    if descriptor_key not in VALID_DESCRIPTORS:
        print(f"Invalid descriptor '{descriptor_key}'. Valid options are: {list(VALID_DESCRIPTORS)}")
        return

    # Validate function column
    if function_column not in df.columns:
        print(f"Missing function column '{function_column}'. Cannot map colors based on this column.")
        return

    # Validate that the function column has numeric data
    if not np.issubdtype(df[function_column].dtype, np.number):
        print(f"Function column '{function_column}' must contain numeric data for color mapping.")
        return

    # Construct descriptor column names for RDKit and PubChem
    desc_prefix = DESCRIPTOR_MAPPING[descriptor_key]
    col_rdkit = f"{desc_prefix}_RDKit"
    col_pubchem = f"{desc_prefix}_PubChem"

    # Check for missing descriptor columns
    missing_cols = [c for c in [col_rdkit, col_pubchem] if c not in df.columns]
    if missing_cols:
        print(f"Missing columns for descriptor '{descriptor_key}': {missing_cols}. Skipping plot.")
        return

    # Select relevant data and drop rows with missing values in descriptor columns or function column
    df_pair = df.dropna(subset=[col_rdkit, col_pubchem, function_column])
    if df_pair.empty:
        print(f"No data available for '{col_rdkit}' and '{col_pubchem}' after dropping NA. Skipping plot.")
        return

    # Extract RDKit and PubChem data
    rdkit_values = df_pair[col_rdkit]
    pubchem_values = df_pair[col_pubchem]
    func_values = df_pair[function_column]

    # Compute differences and statistical measures
    diff = rdkit_values - pubchem_values
    mae = np.mean(np.abs(diff))
    rmse = np.sqrt(np.mean(diff**2))

    # Perform linear regression
    slope, intercept, r_value, p_value, std_err = linregress(rdkit_values, pubchem_values)
    r_squared = r_value**2

    # Prepare colormap for function_column
    func_min = func_values.min()
    func_max = func_values.max()
    norm_func = mcolors.Normalize(vmin=func_min, vmax=func_max)
    cmap = cm.RdBu_r # Red to Blue colormap
    scalar_map = cm.ScalarMappable(norm=norm_func, cmap=cmap)

    # Create the scatter plot
    plt.figure(figsize=(12, 10))
    ax = plt.gca()

    # Scatter plot with color mapping
    scatter = ax.scatter(
        rdkit_values,
        pubchem_values,
        c=func_values,
        cmap=cmap,
        norm=norm_func,
        marker='+',
        s=100,
        alpha=0.7,
        linewidth=1.5,
        label='Data Points'
    )

    # Add colorbar
    cbar = plt.colorbar(scalar_map, ax=ax, orientation='vertical', fraction=0.046, pad=0.04)
    cbar.set_label(function_column, fontsize=18)
    cbar.ax.tick_params(labelsize=16)

    # Plot y = x reference line
    min_val = min(rdkit_values.min(), pubchem_values.min())
    max_val = max(rdkit_values.max(), pubchem_values.max())
    ax.plot(
        [min_val, max_val],
        [min_val, max_val],
        linestyle='--',
        color=GRAPH_COLOR_MAPPING['pair-wise-reflect'],
        label='y = x'
    )

    # Set plot titles and labels
    ax.set_title(f"{desc_prefix}: RDKit vs. PubChem", fontsize=20)
    ax.set_xlabel(f"{desc_prefix}_RDKit", fontsize=18)
    ax.set_ylabel(f"{desc_prefix}_PubChem", fontsize=18)

    # Custom legend entries
    total_label = f'Total Data Points: {len(df_pair)}'
    custom_handle = Line2D([0], [0], marker='+', color='black', linestyle='None',
                           markersize=10, label=total_label)
    y_equals_x_handle = Line2D([0], [0], linestyle='--', color=GRAPH_COLOR_MAPPING['pair-wise-reflect'],
                               label='y = x')

    # Add statistical metrics as text
    stats_text = f"MAE: {mae:.4f}\nRMSE: {rmse:.4f}\n$R^2$: {r_squared:.4f}"
    stats_handle = Line2D([0], [0], color='none', label=stats_text)

    # Compile all legend handles and labels
    handles, labels = ax.get_legend_handles_labels()
    handles.extend([y_equals_x_handle, custom_handle, stats_handle])
    labels.extend(['y = x', total_label, stats_text])

    # Remove duplicate labels
    unique = [(h, l) for h, l in zip(handles, labels) if l not in labels[:handles.index(h)]]
    ax.legend(*zip(*unique), fontsize=14, title="Legend and Metrics", title_fontsize=16, loc='upper left')

    # Add grid
    ax.grid(True, color=GRAPH_COLOR_MAPPING['grid-lines'], linestyle='--', linewidth=0.5)

    # Finalize layout and save the plot
    plt.tight_layout()

    # Ensure the comparisons directory exists
    os.makedirs(comparisons_dir, exist_ok=True)

    # Define file paths
    compare_filename = f"{descriptor_key}_RDKit_vs_PubChem.png"
    compare_path = os.path.join(comparisons_dir, compare_filename)
    info_name = f"{descriptor_key}_RDKit_vs_PubChem_info.txt"
    info_path = os.path.join(comparisons_dir, info_name)

    # Save the scatter plot
    plt.savefig(compare_path, dpi=300)
    plt.close()
    print(f"Comparison scatter saved => {compare_path}")

    # Write analysis info to _info.txt
    with open(info_path, "w") as f:
        f.write(f"Descriptor Analysis Information\n")
        f.write(f"--------------------------------\n")
        f.write(f"Descriptor: {desc_prefix}\n")
        f.write(f"RDKit Column: {col_rdkit}\n")
        f.write(f"PubChem Column: {col_pubchem}\n")
        f.write(f"Function Column for Color Mapping: {function_column}\n")
        f.write(f"Data Points Used: {len(df_pair)}\n")
        f.write(f"Mean Absolute Error (MAE): {mae:.4f}\n")
        f.write(f"Root Mean Squared Error (RMSE): {rmse:.4f}\n")
        f.write(f"R^2 Value: {r_squared:.4f}\n")
    print(f"Comparison info saved => {info_path}")

def classify_fragments(df_in: pd.DataFrame, tag: str) -> pd.DataFrame:
    """
    Classifies molecular fragments based on 'Total_count' and 'ratio', 
    assigning them as BBB+ or BBB-.

    - Filters fragments with at least 3 occurrences.
    - Selects fragments where 'ratio' is below 0.35 (BBB+) or above 0.65 (BBB-).
    - Discards fragments that do not meet filtering criteria.

    Parameters:
        df_in (pd.DataFrame): Input DataFrame containing fragment data.
        tag (str): Fragment type identifier (e.g., 'BRICS').

    Returns:
        pd.DataFrame: Classified DataFrame with an added 'BBB+/BBB-' column.
    """
    # Return the input DataFrame if empty
    if df_in.empty:
        return df_in
    
    # Ensure required columns exist in the DataFrame
    req_cols = ["Total_count", "ratio"]
    for rc in req_cols:
        if rc not in df_in.columns:
            print(f"{tag}: Missing '{rc}' column. Skipping classification.")
            return df_in

    # Filter fragments with at least 3 occurrences
    df_out = df_in.loc[df_in["Total_count"] >= 3].copy()
    
    # Apply ratio-based filtering (retain only strong BBB+ or BBB- classifications)
    df_out = df_out.loc[(df_out["ratio"] < 0.35) | (df_out["ratio"] > 0.65)]
    
    # Return if no fragments remain after filtering
    if df_out.empty:
        print(f"{tag}: No rows remain after ratio/Total_count filter.")
        return df_out

    # Assign BBB+/BBB- labels based on ratio threshold
    df_out["BBB+/BBB-"] = np.where(df_out["ratio"] < 0.35, "BBB+", "BBB-")
    return df_out

def analyze_group(df_group: pd.DataFrame, group_name: str, parent_dir: str):
    """
    Analyzes a molecular group by generating histograms and 
    RDKit vs. PubChem descriptor comparisons.

    - Creates output directories for histograms and comparisons.
    - Generates histograms of descriptor distributions within BBB+/BBB- categories.
    - Performs RDKit vs. PubChem scatter comparisons for each descriptor.

    Parameters:
        df_group (pd.DataFrame): DataFrame containing the group's data.
        group_name (str): Name of the group (e.g., 'molecules', 'brics').
        parent_dir (str): Parent directory where group folders will be stored.

    Returns:
        None: Saves histograms and comparison plots in corresponding directories.
    """
    # Return if the DataFrame is empty
    if df_group.empty:
        return
    
    # Define group folder paths
    group_folder = os.path.join(parent_dir, group_name)
    os.makedirs(group_folder, exist_ok=True)

    grouping_folder = os.path.join(group_folder, "grouping")
    os.makedirs(grouping_folder, exist_ok=True)

    comparisons_folder = os.path.join(group_folder, "comparisons")
    os.makedirs(comparisons_folder, exist_ok=True)

    # Iterate over all valid descriptors
    for dkey in VALID_DESCRIPTORS:
        # Generate histograms categorized by BBB+/BBB- status
        plot_histograms(dkey, df_group, "BBB+/BBB-", grouping_folder)

        # Construct descriptor column names
        desc_prefix = DESCRIPTOR_MAPPING[dkey]
        rd_col = f"{desc_prefix}_RDKit"
        pub_col = f"{desc_prefix}_PubChem"

        # If both RDKit and PubChem columns exist, generate comparison plots
        if rd_col in df_group.columns and pub_col in df_group.columns:
            compare_and_plot_rdkit_vs_pubchem(dkey, df_group, comparisons_folder)

def main(parent_dir):
    """
    Main function for analyzing molecular data.

    - Detects and loads CSV files from the specified directory.
    - Classifies fragments (BRICS, RINGS, SIDECHAINS) based on BBB permeability.
    - Performs statistical analysis and visualization for each category.
    - Outputs results to 'grouping' and 'comparisons' subdirectories.

    Parameters:
        parent_dir (str): Directory containing molecule-related CSV files.

    Returns:
        None: Saves results and visualizations to disk.
    """
    
    # Set default directory if none is provided
    if not parent_dir:
        parent_dir = os.getcwd()
        
    # Validate directory existence
    if not os.path.isdir(parent_dir):
        print(f"Directory '{parent_dir}' not found. Exiting.")
        exit(1)

    # Initialize file paths
    molecules_csv   = None
    brics_csv       = None
    rings_csv       = None
    sidechains_csv  = None

    # Detect relevant CSV files in the directory
    for fname in os.listdir(parent_dir):
        fpath = os.path.join(parent_dir, fname)
        if os.path.isfile(fpath):
            if fname.endswith("_molecules.csv"):
                molecules_csv = fpath
            elif fname.endswith("_brics.csv"):
                brics_csv = fpath
            elif fname.endswith("_rings.csv"):
                rings_csv = fpath
            elif fname.endswith("_sidechains.csv"):
                sidechains_csv = fpath

    # Load CSV files into DataFrames
    df_molecules  = pd.DataFrame()
    df_brics      = pd.DataFrame()
    df_rings      = pd.DataFrame()
    df_sidechains = pd.DataFrame()

    # Load molecules CSV if found
    if molecules_csv and os.path.exists(molecules_csv):
        print(f"Found molecules CSV: {molecules_csv}")
        df_molecules = pd.read_csv(molecules_csv)
    else:
        print("No molecules CSV found. Skipping molecules.")

    # Load BRICS CSV if found
    if brics_csv and os.path.exists(brics_csv):
        print(f"Found BRICS CSV: {brics_csv}")
        df_brics = pd.read_csv(brics_csv)
    else:
        print("No brics CSV found. Skipping brics.")

    # Load rings CSV if found
    if rings_csv and os.path.exists(rings_csv):
        print(f"Found rings CSV: {rings_csv}")
        df_rings = pd.read_csv(rings_csv)
    else:
        print("No rings CSV found. Skipping rings.")

    # Load sidechains CSV if found
    if sidechains_csv and os.path.exists(sidechains_csv):
        print(f"Found sidechains CSV: {sidechains_csv}")
        df_sidechains = pd.read_csv(sidechains_csv)
    else:
        print("No sidechains CSV found. Skipping sidechains.")

    # Classify BRICS, RINGS, and SIDECHAINS
    df_brics      = classify_fragments(df_brics, "BRICS")
    df_rings      = classify_fragments(df_rings, "RINGS")
    df_sidechains = classify_fragments(df_sidechains, "SIDECHAINS")

    # Analyze each group
    print("\n=== Analyzing Molecules ===")
    analyze_group(df_molecules, "molecules", parent_dir)

    print("\n=== Analyzing BRICS ===")
    analyze_group(df_brics, "brics", parent_dir)

    print("\n=== Analyzing RINGS ===")
    analyze_group(df_rings, "rings", parent_dir)

    print("\n=== Analyzing SIDECHAINS ===")
    analyze_group(df_sidechains, "sidechains", parent_dir)

    print("\nAll done. Please check the 'grouping' and 'comparisons' subfolders within each group folder.")


if __name__ == "__main__":
    """
    Command-line interface for running molecule and fragment analysis.

    - Parses command-line arguments for specifying the parent directory.
    - If no argument is provided, prompts the user for a directory.
    - Calls 'main()' to execute analysis.
    """
    # Initialize argument parser
    parser = argparse.ArgumentParser(description="Generate analysis of molecules from preprocessing.ext.py.")
    
    # Define command-line argument
    parser.add_argument("--parent_dir", help="Path to parent directory with generated files.")
    
    # Parse arguments
    args = parser.parse_args()

    # Execute main function with provided or user-specified directory
    if args.parent_dir:
        main(args.parent_dir)
    else:
        parent_dir = input("Enter directory with the CSV files (default data/B3DB_processed/): ").strip()
        main(parent_dir if parent_dir else "data/B3DB_processed/")