"""
Author: Ben Franey
Version: 5.2.4 - Publish: 1.0
Last Review Date: 29-01-2025
Overview:
Processes a CSV of molecular data (B3DB_full.csv), computes RDKit descriptors, optionally 
retrieves PubChem properties, and extracts molecular fragments (BRIC, RING, SIDE_CHAIN). 
Outputs structured JSON and CSV files with molecule- and fragment-level descriptors.

Key Features:
- Computes RDKit 2D/3D descriptors: MW, LogP, HBA, HBD, TPSA, charge, bond types, 
  Wiener Index, Eccentric Connectivity Index, and Radius of Gyration.
- Optional PubChem descriptor retrieval with retry handling.
- Extracts and analyzes molecular fragments:
  - Computes 2D descriptors for BRIC, RING, and SIDE_CHAIN fragments.
  - Tracks BBB+ and BBB- occurrence, normalizing counts.
- Merges RDKit and PubChem descriptors with averaged properties.
- Generates structured outputs:
  - JSON: Full descriptor dataset.
  - CSV: Molecule- and fragment-level summaries, statistical reports.
  - Error logs highlighting missing data.
- Supports CLI execution or interactive input if no arguments are provided.

Usage example:
python src/preprocess.py \
    --input_csv input.csv \
    --output_json output.json \
    --use_pubchem n \
    --calculate_fragment_properties n
        
--input_csv: Path to the input CSV file.
--output_json: Path to save the processed JSON file.
--use_pubchem: Whether to retrieve PubChem properties (y for yes, n for no, default: n).
--calculate_fragment_properties: Whether to calculate fragment properties (y for yes, n for no, default: n
"""

import os
import re
import math
import json
import time
import argparse
import numpy as np
import pandas as pd
import requests
import pubchempy as pcp
from typing import Optional
from tqdm import tqdm
from rdkit import Chem
from rdkit.Chem import (
    AllChem,
    Descriptors,
    rdMolDescriptors,
    BRICS
)
from rdkit.Chem.rdmolops import GetAdjacencyMatrix
import networkx as nx
from collections import defaultdict

# RDKit Descriptors
VALID_DESCRIPTORS = {
    'mw',
    'logp',
    'tpsa',
    'hbond_acceptors',
    'hbond_donors',
    'flexibility',
    'charge',
    'atom_stereo',
    'atom_count_c',
    'atom_count_h',
    'atom_count_o',
    'atom_count_n',
    'atom_count_s',
    'bond_count_single',
    'bond_count_double',
    'bond_count_triple',
    'bond_count_aromatic',
    'wiener_index',
    'eccentric_connectivity_index'
}

# PubChem Descriptors
PUBCHEM_DESCRIPTOR_MAPPING = {
    "logp": "xlogp",
    "tpsa": "tpsa",
    "hbond_acceptors": "h_bond_acceptor_count",
    "hbond_donors": "h_bond_donor_count",
    "flexibility": "rotatable_bond_count",
    "charge": "charge",
    "atom_stereo": "atom_stereo_count",
}


def get_source_from_column(col: str) -> str:
    """
    Extracts the data source from a column name formatted as 'Property_Source'.
    Interprets column names such as 'MW_RDKit' or 'MW_PubChem' as 'RDKit' or 'PubChem', respectivly.
    
    Parameters:
        col (str): Column name formatted as 'Property_Source'. If no underscore is present, returns the input as is.
    
    Returns:
        str: The extracted source name if an underscore is found.
        str: The original column name if no underscore is present.
    """

    # Split column name by underscore
    parts = col.split("_")  

    # Return last element as source name if underscore found
    if len(parts) > 1:
        return parts[-1]  
    
    # Return column name of underscore not found
    return col  


def log_debug(message: str):
    print(f"[DEBUG] {message}")


def get_pubchem_descriptor(
    smiles: str,
    descriptor: str,
    max_retries: int = 1,
    backoff_factor: float = 1.0
) -> Optional[float]:
    """
    Retrieves a specified molecular descriptor from PubChem using the PubChemPy library with retry logic.
    
    - Supports only specific mapped descriptors (PUBCHEM_DESCRIPTOR_MAPPING)
    - Implements exponential backoff for handling network failures.
    
    Parameters:
        smiles (str): SMILES representation of the molecule.
        descriptor (str): Name of the molecular descriptor to retrieve.
        max_retries (int): Maximum number of retry attempts in case of request failure.
        backoff_factor (float): Multiplier for exponential backoff (default is 1.0).

    Returns:
        Optional[float]: The retrieved descriptor value if found.
        None: if the descriptor is unavailable or the request fails.
    
    Raises:
        ValueError: If the descriptor is not supported or not found for the given SMILES.
        pcp.PubChemHTTPError: If a PubChem-specific HTTP error occurs.
        requests.exceptions.RequestException: If a network-related error occurs.
        Exception: If any other unexpected error occurs.
    """
    attempt = 0
    while attempt < max_retries:
    
        try:
            # Validate that the descriptor is supported
            descriptor_lower = descriptor.lower()
            if descriptor_lower not in PUBCHEM_DESCRIPTOR_MAPPING:
                raise ValueError(f"Descriptor '{descriptor}' not supported for PubChem.")

            # Attempt to retrieve compound information from PubChem
            compounds = pcp.get_compounds(smiles, namespace="smiles")
            if compounds and len(compounds) > 0:
                # Asign compound as first-found compound
                compound = compounds[0]
                
                # Get descriptor of compound
                attr_name = PUBCHEM_DESCRIPTOR_MAPPING[descriptor_lower]
                value = getattr(compound, attr_name, None)
                
                # Check compound descriptor exists
                if value is not None:
                    # Log and return found compound value
                    log_debug(f"PubChem Descriptor '{descriptor}' for '{smiles}': {value}")
                    return value
                
            # If descriptor not found, treat it as a failure to retrieve
            log_debug(f"PubChem descriptor '{descriptor}' not found for '{smiles}'.")
            raise ValueError(f"Descriptor '{descriptor}' not found for SMILES '{smiles}'.")

        # Deal with PubChem error
        except (pcp.PubChemHTTPError, requests.exceptions.RequestException) as e:
            # Exponential backoff for retry
            attempt += 1
            wait_time = backoff_factor * (2 ** (attempt - 1))
            print(f"PubChem request failed on attempt {attempt}/{max_retries}: {e}")
            if attempt < max_retries:
                # Retry, when attempts less than max
                print(f"Retrying in {wait_time} seconds...")
                time.sleep(wait_time)
            else:
                # Failed, skip molecule property
                print(f"Failed after {max_retries} attempts for PubChem descriptor '{descriptor}'. Skipping.")
                raise e
        
        # General Exception handling
        except Exception as e:
            print(f"Error fetching PubChem descriptor '{descriptor}' for SMILES '{smiles}': {e}")
            raise e


def is_valid_smiles(smiles: str) -> bool:
    """
    Validates a given SMILES string in RDKit.

    Parameters:
        smiles (str): SMILES representation of the molecule.

    Returns:
        bool: True/False if the SMILES is valid/invalid.
    """
    mol = Chem.MolFromSmiles(smiles)
    return mol is not None


def calculate_wiener_index(mol: Chem.Mol) -> float:
    """
    Computes the Wiener index for a given molecular structure.

    - The Wiener index is computed from the adjacency matrix.
    - Uses NetworkX to determine shortest path lengths in the molecular graph.
    - Assumes an unweighted molecular graph (all bond types treated equally).
    - Only applicable to fully connected molecular structures.

    Parameters:
        mol (Chem.Mol): RDKit molecule object.

    Returns:
        float: The computed Wiener index.

    Raises:
        ValueError: If the adjacency matrix cannot be generated.
    """
    # Generate adjacency matrix
    adj_matrix = GetAdjacencyMatrix(mol)
    
    # Create a graph from the adjacency matrix
    G = nx.Graph(adj_matrix)
    
    # Compute shortest path lengths for all node pairs
    path_lengths = dict(nx.all_pairs_shortest_path_length(G))
    
    # Compute and return Wiener index as half the sum of all shortest path distances
    wiener_index = 0.0
    for node, lengths in path_lengths.items():
        wiener_index += sum(lengths.values())
    return wiener_index / 2


def eccentric_connectivity_index(mol: Chem.Mol) -> float:
    """
    Computes the eccentric connectivity index using NetworkX.

    - Calculates node eccentricity and degree from the molecular adjacency matrix.
    - Returns 0.0 if the eccentricity computation fails.

    Parameters:
        mol (Chem.Mol): RDKit molecule object.

    Returns:
        float: The computed eccentric connectivity index.
    """
    # Generate adjacency matrix of the molecular graph
    adj_matrix = GetAdjacencyMatrix(mol)
    
    # Construct a graph from the adjacency matrix
    G = nx.Graph(adj_matrix)
    
    # Compute eccentricity for each node (max shortest path length from the node)
    try:
        eccentricity = nx.eccentricity(G)
    # Handle errors in eccentricity calculation, returning 0.0
    except nx.NetworkXError as e:
        log_debug(f"Eccentricity calculation error: {e}")
        return 0.0
    
    # Compute node degrees
    degrees = dict(G.degree())
    
    # Compute eccentric connectivity index: sum of (eccentricity * degree) for all nodes
    return sum(eccentricity[node] * degrees[node] for node in G.nodes())


def count_specific_atoms(mol: Chem.Mol, atom_list: list, aromatic: bool = None) -> dict:
    """
    Returns a dictionary of counts for each atom in 'atom_list', including implicit hydrogens.

    - Counts explicit and implicit hydrogen atoms.
    - Filters atoms based on aromaticity if not None.

    Parameters:
        mol (Chem.Mol): RDKit molecule object.
        atom_list (list): List of atom symbols (e.g., ['C', 'H', 'O']) to count.
        aromatic (bool): Optional; if specified, counts only atoms with matching aromaticity.

    Returns:
        dict: Dictionary of atom counts.
    """
    
    # Initialize atom counts dictionary with all specified elements set to 0
    counts = {element: 0 for element in atom_list}
    
    # Iterate through all atoms in the molecule
    for atom in mol.GetAtoms():
        symbol = atom.GetSymbol()
        
        # Count explicit and implicit hydrogens for 'H'
        if 'H' in atom_list and symbol != 'H':
            # Adds implicit hydrogens to count
            counts['H'] += atom.GetTotalNumHs()

        # Count the current atom based on the aromatic filter
        if symbol in atom_list:
            if aromatic is None or atom.GetIsAromatic() == aromatic:
                counts[symbol] += 1

    # Return dictionary of counted atoms
    return counts


def count_bond_types(mol: Chem.Mol) -> dict:
    """
    Returns a dictionary with counts for each bond type: Single, Double, Triple, Aromatic.

    - Iterates through all bonds in the molecule.
    - Categorizes bonds into Single, Double, Triple, or Aromatic.
    - Returns bond counts as a dictionary.

    Parameters:
        mol (Chem.Mol): RDKit molecule object.

    Returns:
        dict: Dictionary with bond type counts.
            Single (int): Count of Single chiral centers bonds.
            Double (int): Count of Double bonds.
            Triple (int): Count of Tripple bonds.
            Aromatic (int): Count of Aromatic bonds.
        
    """
    # Initialize bond counts dictionary
    bond_counts = {'Single': 0, 'Double': 0, 'Triple': 0, 'Aromatic': 0}
    
    # Iterate through all bonds in the molecule
    for bond in mol.GetBonds():
        
        # Classify bond type and update counts
        btype = bond.GetBondType()
        if btype == Chem.rdchem.BondType.SINGLE:
            bond_counts['Single'] += 1
        elif btype == Chem.rdchem.BondType.DOUBLE:
            bond_counts['Double'] += 1
        elif btype == Chem.rdchem.BondType.TRIPLE:
            bond_counts['Triple'] += 1
        elif btype == Chem.rdchem.BondType.AROMATIC:
            bond_counts['Aromatic'] += 1
            
    # Return dictionary of bond counts
    return bond_counts


def count_chiral_centers(mol: Chem.Mol) -> dict:
    """
    Assigns stereochemistry and returns a dictionary with total R and S chiral centers.

    - Assigns stereochemistry to the molecule.
    - Identifies and counts chiral centers with assigned configurations (R/S).
    - Handles exceptions and logs errors if stereochemistry assignment fails.

    Parameters:
        mol (Chem.Mol): RDKit molecule object.

    Returns:
        dict: Dictionary containing counts of total chiral centers, R isomers, and S isomers.
            Total_Chiral_Centers (int): Count of total chiral centers.
            R_Isomers (int): Count of R Isomers.
            S_Isomers (int): Count of S Isomers.
    """
    # Assign stereochemistry to the molecule
    try:
        Chem.AssignStereochemistry(mol, cleanIt=True, force=False)
        
        # Identify chiral centers, excluding unassigned ones
        chiral_centers = Chem.FindMolChiralCenters(mol, includeUnassigned=False)
    
    # Handle errors, log error and return empty chiral center list
    except Exception as e:
        log_debug(f"Error assigning stereochemistry: {e}")
        chiral_centers = []
        
    # Count total, R, and S chiral centers
    total_chiral_centers = len(chiral_centers)
    r_count = sum(1 for center in chiral_centers if center[1] == 'R')
    s_count = sum(1 for center in chiral_centers if center[1] == 'S')
    
    # Return dictionary of stereochemistry
    return {
        'Total_Chiral_Centers': total_chiral_centers,
        'R_Isomers': r_count,
        'S_Isomers': s_count
    }


def count_EZ_isomers(mol: Chem.Mol) -> dict:
    """
    Counts E and Z isomers for double bonds in the molecule.

    - Identifies double bonds with defined E/Z stereochemistry.
    - Categorizes bonds as E or Z based on assigned stereochemistry.

    Parameters:
        mol (Chem.Mol): RDKit molecule object.

    Returns:
        dict: Dictionary containing counts of E and Z isomers.
            E_Isomers (int): Count of E Isomers.
            Z_Isomers (int): Count of Z Isomers.
    """
    e_count = 0
    z_count = 0
    
    # Iterate through all bonds in the molecule
    for bond in mol.GetBonds():
        # If double bond then check stereochemistry
        if bond.GetBondType() == Chem.rdchem.BondType.DOUBLE:
            stereo = bond.GetStereo()
            
            # Classify the bond as E or Z based on its stereochemistry
            if stereo == Chem.rdchem.BondStereo.STEREOZ:
                z_count += 1
            elif stereo == Chem.rdchem.BondStereo.STEREOE:
                e_count += 1

    # Return dictionary of stereochemistry 
    return {'E_Isomers': e_count, 'Z_Isomers': z_count}


def count_stereocenters(mol):
    """
    Counts potential stereocenters in a molecule.

    - Includes chiral centers (atoms with four distinct substituents).
    - Includes E/Z isomers (double bonds with distinct substituents).
    
    Parameters:
        mol (Chem.Mol): RDKit molecule object.

    Returns:
        tuple:
            int: Number of chiral centers.
            int: Number of E/Z stereoisomeric double bonds.
    """
    
    # Identify chiral centers, including unassigned ones
    chiral_centers = Chem.FindMolChiralCenters(
        mol, includeUnassigned=True, useLegacyImplementation=False
    )
    num_chiral_centers = len(chiral_centers)
    
    num_stereo_double_bonds = 0
    # Iterate through all bonds to identify double bonds with stereochemistry
    for bond in mol.GetBonds():
        if bond.GetBondType() == Chem.BondType.DOUBLE and bond.GetStereo() != Chem.BondStereo.STEREONONE:
            begin_atom = bond.GetBeginAtom()
            end_atom = bond.GetEndAtom()
            # Check for distinct substituents on both ends of the double bond
            if len(begin_atom.GetNeighbors()) > 1 and len(end_atom.GetNeighbors()) > 1:
                num_stereo_double_bonds += 1
    
    # Return tuple
    return num_chiral_centers, num_stereo_double_bonds


def count_rings_by_size_and_aromaticity(mol: Chem.Mol, ring_sizes: list) -> dict:
    """
    Returns a dictionary keyed by ring size, containing counts of aromatic and non-aromatic rings.

    - Extracts ring information from the molecular structure.
    - Identifies and classifies rings based on size and aromaticity.
    - Tracks the total count of rings for each specified size.

    Parameters:
        mol (Chem.Mol): RDKit molecule object.
        ring_sizes (list): List of ring sizes (integers) to count.

    Returns:
        dict: Dictionary with ring size as keys, each containing:
            aromatic (int): Count of aromatic rings of the given size.
            non_aromatic (int): Count of non-aromatic rings of the given size.
            total (int): Total count of rings of the given size.
    """
    # Retrieve ring information from the molecule
    ring_info = mol.GetRingInfo()
    
    # Initialize dictionary to store ring counts
    ring_counts = {size: {'aromatic': 0, 'non_aromatic': 0, 'total': 0} for size in ring_sizes}
    
    # Iterate through each ring in the molecule
    for ring_atom_indices in ring_info.AtomRings():
        size = len(ring_atom_indices)
        
        # Process only rings of specified sizes
        if size in ring_sizes:
            # Determine if all atoms in the ring are aromatic
            is_aromatic = all(mol.GetAtomWithIdx(idx).GetIsAromatic() for idx in ring_atom_indices)
            ring_counts[size]['total'] += 1
            
            # Categorize as aromatic or non-aromatic
            if is_aromatic:
                ring_counts[size]['aromatic'] += 1
            else:
                ring_counts[size]['non_aromatic'] += 1
    
    # Return dictionary of ring size and aromaticity counts
    return ring_counts


def calculate_radius_of_gyration(mol: Chem.Mol):
    """
    Computes the radius of gyration from the 3D conformer.

    - Requires a valid 3D conformer to compute the radius.
    - Calculates the molecular centroid.
    - Computes the mean squared distance of all atoms from the centroid.
    - Returns the square root of the mean squared distance.
    - Not Mass Weighted radius of gyration.

    Parameters:
        mol (Chem.Mol): RDKit molecule object with a 3D conformer.

    Returns:
        float: Radius of gyration if a 3D conformer is present.
        None: If no 3D conformer is found.
    """
    # Check if the molecule has at least one 3D conformer
    if not mol.GetNumConformers():
        # Return None if no conformer is available
        return None
    
    # Retrieve the first conformer
    conf = mol.GetConformer()
    
    # Extract atomic coordinates as a NumPy array
    coords = np.array(conf.GetPositions(), dtype=float)

    # Compute the centroid of the molecule
    centroid = coords.mean(axis=0)
    
    # Compute mean squared distance of atoms from the centroid
    rg_sq = ((coords - centroid)**2).sum(axis=1).mean()
    return math.sqrt(rg_sq)


def compute_2d_descriptors_rdkit(mol: Chem.Mol) -> dict:
    """
    Compute standard RDKit 2D descriptors - keys end with '_RDKit'.

    - Computes flexibility, molecular weight, logP, HBA, HBD, TPSA, charge, and heavy atom count.
    - Assigns stereochemistry and counts atom and double bond stereoisomers.
    - Counts individual atoms (C, H, O, N, S, etc.).
    - Computes bond types and chirality-related descriptors.
    - Extracts ring counts categorized by size and aromaticity.
    - Computes Wiener and eccentric connectivity indices.

    Parameters:
        mol (Chem.Mol): RDKit molecule object.

    Returns:
        dict: Dictionary containing computed 2D descriptors.
    """
    
    descriptor_dictionary = {}
    
    # Flexibility
    descriptor_dictionary["Flexibility_RDKit"] = Descriptors.NumRotatableBonds(mol)
    # MW
    descriptor_dictionary["MW_RDKit"] = round(Descriptors.MolWt(mol), 4)
    # logP
    descriptor_dictionary["LogP_RDKit"] = round(Descriptors.MolLogP(mol), 4)
    # HBA
    descriptor_dictionary["HBA_RDKit"] = Descriptors.NumHAcceptors(mol)
    # HBD
    descriptor_dictionary["HBD_RDKit"] = Descriptors.NumHDonors(mol)
    # TPSA
    descriptor_dictionary["TPSA_RDKit"] = round(Descriptors.TPSA(mol), 4)
    # Charge
    descriptor_dictionary["Charge_RDKit"] = Chem.GetFormalCharge(mol)
    # Heavy atom
    descriptor_dictionary["HeavyAtom_RDKit"] = Descriptors.HeavyAtomCount(mol)

    # Compute atom and double bond stereochemistry
    try:
        # Add explicit hydrogens to ensure proper stereochemistry assignment
        temp_mol_h = Chem.AddHs(mol)
        
        # Force stereochemistry assignment
        Chem.AssignStereochemistry(temp_mol_h, force=True, cleanIt=True)
        
        # Count stereocenters
        chiral_count, double_bond_count = count_stereocenters(temp_mol_h)
        
        # Store total atom stereoisomers
        descriptor_dictionary["AtomStereo_RDKit"] = chiral_count + double_bond_count

    # Handle errors and set default value
    except Exception as e:
        descriptor_dictionary["AtomStereo_RDKit"] = None
        log_debug(f"Stereo assignment error (2D descriptors RDKit): {e}")
        print(f"Error calculating stereochemistry: {e}")


    # Count atoms (common elements)
    all_atoms_list = ['C','H','O','N','S','F','Cl','Br','I','P', 'B','Li']
    atom_counts = count_specific_atoms(mol, all_atoms_list)
    
    # Store atom counts in the descriptor dictionary
    for elem, val in atom_counts.items():
        descriptor_dictionary[f"AtomCount_{elem}_RDKit"] = val

    # Count bond types (Single, Double, Triple, Aromatic)
    bond_count = count_bond_types(mol)
    
    # Store bond counts in the descriptor dictionary
    for type, value in bond_count.items():
        descriptor_dictionary[f"BondCount_{type}_RDKit"] = value

    # Chiral centers
    chiral_centers = count_chiral_centers(mol)
    
    # Store chiral centers in the descriptor dictionary
    descriptor_dictionary.update({
        "Total_Chiral_Centers_RDKit": chiral_centers['Total_Chiral_Centers'],
        "R_Isomers_RDKit": chiral_centers['R_Isomers'],
        "S_Isomers_RDKit": chiral_centers['S_Isomers']
    })

    # Count E/Z isomers in the molecule
    ez_isomers = count_EZ_isomers(mol)
    
    # Store E/Z isomers in the descriptor dictionary
    descriptor_dictionary.update({
        "E_Isomers_RDKit": ez_isomers['E_Isomers'],
        "Z_Isomers_RDKit": ez_isomers['Z_Isomers']
    })

    # Count rings based on size and aromaticity
    ring_sizes = [4, 5, 6, 8]
    ring_counts = count_rings_by_size_and_aromaticity(mol, ring_sizes)
    
    total_aromatic = 0
    total_non_aromatic = 0
    total_rings = 0
    
    # Store ring counts in the descriptor dictionary
    for size in ring_sizes:
        descriptor_dictionary[f"Num_{size}_Rings_Aromatic_RDKit"] = ring_counts[size]['aromatic']
        descriptor_dictionary[f"Num_{size}_Rings_NonAromatic_RDKit"] = ring_counts[size]['non_aromatic']
        descriptor_dictionary[f"Num_{size}_Rings_Total_RDKit"] = ring_counts[size]['total']
        
        # Update total ring counts
        total_aromatic += ring_counts[size]['aromatic']
        total_non_aromatic += ring_counts[size]['non_aromatic']
        total_rings += ring_counts[size]['total']

    # Store total ring counts
    descriptor_dictionary["Num_Aromatic_Rings_RDKit"] = total_aromatic
    descriptor_dictionary["Num_NonAromatic_Rings_RDKit"] = total_non_aromatic
    descriptor_dictionary["Num_Total_Rings_RDKit"] = total_rings

    # Compute Wiener and Eccentric Connectivity indices
    descriptor_dictionary["WienerIndex_RDKit"] = round(calculate_wiener_index(mol), 4)
    descriptor_dictionary["EccentricConnectivityIndex_RDKit"] = round(eccentric_connectivity_index(mol), 4)

    # Return computed descriptor dictionary
    return descriptor_dictionary


def compute_3d_descriptors_rdkit(mol: Chem.Mol) -> dict:
    """
    Compute 3D descriptors from RDKit (radius of gyration).
    
    - Requires a valid 3D conformer to compute descriptors.
    - Computes the radius of gyration from atomic coordinates.
    - Keys in the output dictionary end with '_RDKit'.

    Parameters:
        mol (Chem.Mol): RDKit molecule object with a 3D conformer.

    Returns:
        dict: Dictionary containing computed 3D descriptors.
              RadiusOfGyration_RDKit (float): Radius of gyration if a 3D conformer exists.
              None: If no conformer is present.
    """
    # Initialize dictionary with default value (None) for 3D descriptors
    descriptor_dictionary_3d = {
        "RadiusOfGyration_RDKit": None
    }
    
    # Check if the molecule has at least one 3D conformer
    if mol.GetNumConformers() == 0:
        # Return dictionary with default values if no conformer exists
        return descriptor_dictionary_3d
    
    # TO DO: Energy Minimisation, crashed.

    # Compute radius of gyration
    radius_of_gyration = calculate_radius_of_gyration(mol)
    
    # Store computed radius of gyration if valid
    if radius_of_gyration is not None:
        descriptor_dictionary_3d["RadiusOfGyration_RDKit"] = round(radius_of_gyration, 4)

    # Return computed 3D descriptors
    return descriptor_dictionary_3d


def compute_descriptors_pubchem(smiles: str) -> dict:
    """
    Retrieve molecular descriptors from PubChem.

    - Retrieves MW_PubChem, LogP_PubChem, TPSA_PubChem, HBA_PubChem, HBD_PubChem, 
      Flexibility_PubChem, Charge_PubChem, and AtomStereo_PubChem.
    - Fetches descriptors for the entire molecule (not fragments).
    - Uses PubChem's API with retry logic for failed requests.

    Parameters:
        smiles (str): SMILES representation of the molecule.

    Returns:
        dict: Dictionary containing PubChem descriptors.
              - Keys: Descriptor names ending with "_PubChem".
              - Values: Rounded numerical values or None if retrieval fails.
    """
    # Initialize descriptor dictionary with default values (None)
    descriptor_dictionary_pubchem = {
        "LogP_PubChem": None,
        "TPSA_PubChem": None,
        "HBA_PubChem": None,
        "HBD_PubChem": None,
        "Flexibility_PubChem": None,
        "Charge_PubChem": None,
        "AtomStereo_PubChem": None
    }
    
    # Mapping of descriptor names from PubChem to dictionary keys
    pub_mapping = {
        "logp": "LogP_PubChem",
        "tpsa": "TPSA_PubChem",
        "hbond_acceptors": "HBA_PubChem",
        "hbond_donors": "HBD_PubChem",
        "flexibility": "Flexibility_PubChem",
        "charge": "Charge_PubChem",
        "atom_stereo": "AtomStereo_PubChem"
    }
    
    # Iterate over descriptor mappings and retrieve values from PubChem
    for short_name, col_name in pub_mapping.items():
        try:
            # Fetch descriptor value from PubChem
            val = get_pubchem_descriptor(smiles, short_name, max_retries=1, backoff_factor=1.0) # No Retries for brevity
            
            # Store value if valid (numeric type), rounding to 4 decimals
            if val is not None and isinstance(val, (int,float)):
                descriptor_dictionary_pubchem[col_name] = round(val, 4)
                
        # Log retrieval failure and continue with other descriptors
        except Exception as e:
            log_debug(f"PubChem retrieval failed for {short_name} in SMILES '{smiles}': {e}")
            
    # Return dictionary with retrieved descriptors
    return descriptor_dictionary_pubchem


def merge_rdkit_and_pubchem_descriptors(rd: dict, pc: dict) -> dict:
    """
    Merge RDKit and PubChem descriptors, storing the final average as <Name>_Avg
    for the 7 main descriptors: LogP, TPSA, HBA, HBD, Flexibility, Charge, AtomStereo.
    Other RDKit-only descriptors (e.g., 3D, ring counts) are left as-is.

    - Merges RDKit and PubChem descriptor dictionaries.
    - Computes the average value for key descriptors present in both sources.
    - Retains RDKit-only and PubChem-only descriptors unchanged.

    Parameters:
        rd (dict): RDKit descriptors dictionary with keys like "MW_RDKit".
        pc (dict): PubChem descriptors dictionary with keys like "MW_PubChem".

    Returns:
        dict: Merged dictionary with averages for specified descriptors.
    """
    # Initialize merged descriptor dictionary
    merged = {}
    
    # Combine RDKit descriptors into the merged dictionary
    for k, v in rd.items():
        merged[k] = v

    # Combine PubChem descriptors into the merged dictionary
    for k, v in pc.items():
        merged[k] = v

    # Helper function to calculate the average of two values
    def avg2(a, b):
        # Return None if both values are missing
        if a is None and b is None:
            return None
        
        # Use PubChem value if RDKit value is missing
        if a is None:
            return b
        
        # Use RDKit value if PubChem value is missing
        if b is None:
            return a
        
        # Compute rounded average
        return round((a + b) / 2, 4)

    # Main descriptors to average
    props = ["LogP", "TPSA", "HBA", "HBD", "Flexibility", "Charge", "AtomStereo"]

    for p in props:
        rdkit_key = f"{p}_RDKit"
        pubchem_key = f"{p}_PubChem"
        avg_key = f"{p}_Avg"

        # Get RDKit and PubChem values
        v1 = merged.get(rdkit_key, None)
        v2 = merged.get(pubchem_key, None)
        
        # Compute and store the averaged value
        merged[avg_key] = avg2(v1, v2)

    # Return dictionary with merged descriptors
    return merged


def compute_fragment_2d_descriptors(smiles_str: str, remove_brics_placeholder=False) -> dict:
    """
    Computes 2D descriptors for a molecular fragment (BRIC, RING, or SIDE_CHAIN).

    - Optionally removes BRICS placeholders from the SMILES string.
    - Generates an RDKit molecule from the cleaned SMILES.
    - Computes RDKit 2D descriptors (no PubChem descriptors included).
    - Returns the computed descriptors.

    Parameters:
        smiles_str (str): SMILES representation of the fragment.
        remove_brics_placeholder (bool): If True, removes BRICS placeholders.

    Returns:
        dict: Dictionary containing computed RDKit 2D descriptors.
              - Empty dictionary if SMILES conversion fails.
    """
    # Initialize cleaned SMILES string
    cleaned_smiles = smiles_str
    
    # Remove BRICS placeholders
    if remove_brics_placeholder:
        # Matches (\[digits*\]) or [digits*\] or leading/trailing '='
        cleaned_smiles = re.sub(r'^\=|\=$|(\(\[\d+\*\]\)|\[\d+\*\])', '', cleaned_smiles)

    try:
        # Convert SMILES to RDKit molecule
        mol = Chem.MolFromSmiles(cleaned_smiles)
        
        # Return empty dictionary if conversion fails
        if mol is None:
            return {}
        
    # Log error and return empty dictionary
    except Exception as e:
        log_debug(f"Error in compute_fragment_2d_descriptors: {e}")
        return {}

    # Compute RDKit 2D descriptors only, no PubChem for brevity
    rd2 = compute_2d_descriptors_rdkit(mol)
    
    # Return computed 2D descriptors
    return rd2


def process_ring_side_chains(smiles):
    """
    Extracts ring fragments and side-chain fragments from a molecule SMILES.

    - Identifies ring fragments using RDKit's ring detection.
    - Extracts side chains by finding atoms outside of ring systems.
    - Returns two lists: one for ring SMILES and another for side-chain SMILES.

    Parameters:
        smiles (str): SMILES representation of the molecule.

    Returns:
        tuple:
            - list: SMILES strings of ring fragments.
            - list: SMILES strings of side-chain fragments.
    """
    rings_list = []
    side_chains_list = []
    
    try:
        # Convert SMILES to RDKit molecule
        mol = Chem.MolFromSmiles(smiles)
        
        # Return empty lists if conversion fails
        if mol is None:
            return rings_list, side_chains_list

        # Get ring information from the molecule
        ring_info = mol.GetRingInfo()
        
        # Retrieve atom indices for each ring
        rings = ring_info.AtomRings()
        
        # Extract ring fragments as SMILES
        for ring in rings:
            ring_sm = Chem.MolFragmentToSmiles(mol, atomsToUse=ring)
            rings_list.append(ring_sm)
        
        # Identify atoms that belong to rings
        ring_atoms = set(a for ring in rings for a in ring)
       
        # Identify non-ring (side-chain) atoms
        non_ring_atoms = [
            atom.GetIdx()
            for atom in mol.GetAtoms()
            if atom.GetIdx() not in ring_atoms
        ]
        
        # Track atoms already assigned to a fragment
        visited_atoms = set()
        
        # Extract side-chain fragments
        for atom_idx in non_ring_atoms:
            if atom_idx not in visited_atoms:
                # Identify connected atoms that are not part of a ring
                neighbors = [atom_idx] + [
                    nbr.GetIdx()
                    for nbr in mol.GetAtomWithIdx(atom_idx).GetNeighbors()
                    if nbr.GetIdx() not in ring_atoms and nbr.GetIdx() not in visited_atoms
                ]
                
                # Extract fragment SMILES
                fragment = Chem.MolFragmentToSmiles(mol, atomsToUse=neighbors)
                side_chains_list.append(fragment)
                
                # Mark atoms as visited
                visited_atoms.update(neighbors)
                
    # Log any errors encountered during processing
    except Exception as e:
        log_debug(f"process_ring_side_chains error: {e}")

    # Return extracted ring and side-chain SMILES lists
    return rings_list, side_chains_list


def check_for_missing_data(
    df: pd.DataFrame,
    type_name: str,
    id_col: str,
    error_log: list,
    property_wise_log: list,
    smiles_col: str = None
):
    """
    Checks numeric columns in df for missing data and logs them appropriately.

    - Iterates through all numeric columns in the dataframe.
    - Identifies missing (NaN) values in numeric columns.
    - Logs missing data in an error log list as formatted strings.
    - Stores detailed missing data records in a property-wise log.

    Parameters:
        df (pd.DataFrame): Input dataframe containing descriptor data.
        type_name (str): Label for the type of data being checked.
        id_col (str): Column name used to identify rows (e.g., molecule ID).
        error_log (list): List to store formatted error messages.
        property_wise_log (list): List to store structured missing data logs.
        smiles_col (str, optional): Column containing SMILES strings (default: None).

    Returns:
        None: Logs missing data but does not return a modified dataframe.
    """
    # Identify numeric columns in the dataframe
    numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    
    # Iterate through each row in the dataframe
    for idx, row in df.iterrows():
        row_id = row.get(id_col, None)
        row_smiles = row.get(smiles_col, None) if (smiles_col and smiles_col in df.columns) else "N/A"
        
        # Check each numeric column for missing values
        for col in numeric_cols:
            val = row[col]
            if pd.isna(val):
                # Log missing data in error log
                error_log.append(f"[{type_name}] Missing data for {col} in {id_col}='{row_id}'")
                
                # Store detailed missing data information in property-wise log
                property_wise_log.append({
                    "Type": type_name,
                    "ID_Column": id_col,
                    "ID_Value": row_id,
                    "Descriptor": col,
                    "SMILES": row_smiles
                })


class FragmentStore:
    """
    Holds unique fragments (BRIC, RING, or SIDE_CHAIN).

    - Stores fragments uniquely using their SMILES representation as keys.
    - Tracks the number of BBB+ and BBB- molecules that contain each fragment.
    - Computes and stores RDKit 2D descriptors for each fragment (no PubChem descriptors).

    Attributes:
        data (dict): Stores fragment information indexed by SMILES.
        counter (int): Unique ID counter for new fragments.
        prefix (str): Prefix used for fragment IDs.
        bbb_plus_count (int): Total count of BBB+ molecules.
        bbb_minus_count (int): Total count of BBB- molecules.
    """
    def __init__(self, prefix="X"):
        """
        Initializes the fragment store.

        Parameters:
            prefix (str): Prefix for fragment IDs (default: "X").
        """
        self.data = {}
        self.counter = 1
        self.prefix = prefix
        self.bbb_plus_count = 0
        self.bbb_minus_count = 0

    def add_fragment(self, smiles: str, is_bbb_plus: bool, remove_brics_placeholder=False):
        """
        Adds a fragment if it is new, then updates its BBB+ / BBB- count.

        - Computes RDKit 2D descriptors for new fragments.
        - Assigns a unique fragment ID.
        - Tracks the number of occurrences in BBB+ and BBB- molecules.

        Parameters:
            smiles (str): SMILES representation of the fragment.
            is_bbb_plus (bool): True if the fragment comes from a BBB+ molecule.
            remove_brics_placeholder (bool): If True, removes BRICS placeholders.

        Returns:
            dict: Updated fragment data.
        """
        
        # Check if fragment already exists in the store
        if smiles not in self.data:
            # Compute RDKit 2D descriptors for the fragment
            desc = compute_fragment_2d_descriptors(smiles, remove_brics_placeholder=remove_brics_placeholder)
            
            # Assign a unique fragment ID
            frag_id = f"{self.prefix}{self.counter:04d}"
            
            # Initialize fragment data
            self.data[smiles] = {
                "ID": frag_id,
                "SMILES": smiles,
                "Total_count": 0,
                "BBB+_count": 0,
                "BBB-_count": 0
            }
            
            # Store computed descriptor values
            for k,v in desc.items():
                self.data[smiles][k] = v

            self.counter += 1

        # Increment counters based on BBB permeability classification
        if is_bbb_plus:
            self.data[smiles]["BBB+_count"] += 1
        else:
            self.data[smiles]["BBB-_count"] += 1
        
        self.data[smiles]["Total_count"] += 1

        # Return updated fragment data
        return self.data[smiles]

    def finalize_counts(self):
        """
        Normalizes BBB+ and BBB- usage counts relative to their totals.

        - Computes the normalized BBB+ and BBB- counts.
        - Computes the ratio of BBB- occurrences to total occurrences.

        Returns:
            None: Updates the stored fragment data in place.
        """
        total_plus = float(self.bbb_plus_count)
        total_minus = float(self.bbb_minus_count)
        
        # Iterate over stored fragments and compute normalized values
        for _, d in self.data.items():
            plus_ct = d["BBB+_count"]
            minus_ct = d["BBB-_count"]
            
            # Compute normalized counts
            if total_plus > 0:
                d["BBB+_normalised"] = plus_ct / total_plus
            else:
                d["BBB+_normalised"] = 0.0
            if total_minus > 0:
                d["BBB-_normalised"] = minus_ct / total_minus
            else:
                d["BBB-_normalised"] = 0.0

            # Compute ratio of BBB- occurrences
            s = d["BBB+_normalised"] + d["BBB-_normalised"]
            if s > 0:
                d["ratio"] = round(d["BBB-_normalised"] / s, 4)
            else:
                d["ratio"] = 0.5

    def all_rows_df(self) -> pd.DataFrame:
        """
        Returns a DataFrame of all unique fragments stored.

        Returns:
            pd.DataFrame: DataFrame containing stored fragment data.
        """
        return pd.DataFrame(self.data.values())


def compute_summary_stats(df: pd.DataFrame, data_type: str) -> pd.DataFrame:
    """
    Computes summary statistics (mean and standard deviation) for numeric columns.

    - Identifies numeric columns in the dataframe.
    - Computes the mean and standard deviation for each numeric column.
    - Drops NaN values before computing statistics.
    - Returns a summary DataFrame with computed metrics.

    Parameters:
        df (pd.DataFrame): Input dataframe containing numerical descriptor data.
        data_type (str): Label describing the type of data being processed.

    Returns:
        pd.DataFrame: Summary dataframe containing:
            - "Type" (str): Data type label.
            - "Metric" (str): Column name.
            - "Mean" (float): Computed mean of the column.
            - "SD" (float): Computed standard deviation of the column.
    """
    rows = []
    
    # Identify numeric columns in the dataframe
    numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    
    # Iterate over numeric columns and compute statistics
    for col in numeric_cols:
        # Remove missing values
        col_data = df[col].dropna()
        if len(col_data) > 0:
            # Compute mean
            mean_val = col_data.mean()
            
            # Compute standard deviation with ddof=1 for sample standard deviation
            std_val = col_data.std(ddof=1) if len(col_data) > 1 else 0.0
            
            # Store computed statistic
            rows.append({
                "Type": data_type,
                "Metric": col,
                "Mean": mean_val,
                "SD": std_val
            })
    
    # Return summary statistics as a DataFrame
    return pd.DataFrame(rows)


def generate_json_from_csv(input_csv: str, output_json: str, use_pubchem=False, calc_frag_props=False):
    """
    Parses the input CSV, computes molecular and (optionally) fragment descriptors, 
    generates JSON output, produces CSV outputs, and creates summary statistics.

    - Reads the input CSV and validates required columns.
    - Computes descriptors for each molecule using RDKit and optionally PubChem.
    - Always extracts BRIC, RING, and SIDE_CHAIN fragments; however, if 
      calc_frag_props is False, fragment properties (e.g. MW, LogP, TPSA, etc.) are not calculated.
      In either case, usage counts are updated in the fragment store for CSV output, but these
      usage count keys are not added to the JSON.
    - Stores descriptor data in JSON and CSV formats.
    - Generates summary statistics and checks for missing data.
    - Saves log files and summary reports.

    Parameters:
        input_csv (str): Path to the input CSV file.
        output_json (str): Path to the output JSON file.
        use_pubchem (bool): Whether to retrieve additional descriptors from PubChem.
        calc_frag_props (bool): Whether to calculate full fragment properties (e.g. MW, LogP, TPSA).  
                                If False, only usage counts are updated (but not output to JSON).
    
    Returns:
        None: Saves processed data as JSON and CSV files.
    """

    # Check if input file exists
    if not os.path.exists(input_csv):
        print(f"Input file '{input_csv}' does not exist.")
        return

    # Read input CSV
    try:
        df = pd.read_csv(input_csv)
    except Exception as e:
        print(f"Error reading input CSV: {e}")
        return

    # Required columns for processing
    required_cols = ['NO.', 'SMILES', 'BBB+/BBB-', 'group']
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        print(f"CSV missing required columns: {missing}")
        return

    # Compute total counts of BBB+ and BBB- molecules
    total_bbb_plus_mols = len(df[df["BBB+/BBB-"] == "BBB+"])
    total_bbb_minus_mols = len(df[df["BBB+/BBB-"] == "BBB-"])

    # Initialize fragment stores
    bric_store = FragmentStore(prefix="BRIC")
    bric_store.bbb_plus_count = total_bbb_plus_mols
    bric_store.bbb_minus_count = total_bbb_minus_mols

    ring_store = FragmentStore(prefix="RING")
    ring_store.bbb_plus_count = total_bbb_plus_mols
    ring_store.bbb_minus_count = total_bbb_minus_mols

    sidechain_store = FragmentStore(prefix="SC")
    sidechain_store.bbb_plus_count = total_bbb_plus_mols
    sidechain_store.bbb_minus_count = total_bbb_minus_mols

    # Initialize JSON storage
    json_data = [{"_preface": "Preface, TBC."}]
    molecule_dicts = []

    # Process each molecule in the dataset
    pbar = tqdm(df.iterrows(), total=df.shape[0], desc="Processing SMILES", unit="molecule")
    for idx, row in pbar:
        no_val      = row.get('NO.', None)
        smiles      = row['SMILES']
        bbb_status  = row.get('BBB+/BBB-', None)
        grp         = row.get('group', None)
        is_bbb_plus = (bbb_status == "BBB+")

        # Validate SMILES
        if not smiles or not is_valid_smiles(smiles):
            log_debug(f"[Row {idx+1}] Invalid SMILES: '{smiles}'. Skipping.")
            continue

        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            log_debug(f"[Row {idx+1}] RDKit parse failed for SMILES: '{smiles}' -> skip.")
            continue

        pbar.set_postfix({"Index": idx})

        # Attempt 3D coordinate generation and descriptor calculation for the molecule
        try:
            mol_h = Chem.AddHs(mol)
            params = AllChem.ETKDGv3()
            params.useRandomCoords = True
            params.maxAttempts = 50
            params.pruneRmsThresh = 0.1
            embed_result = AllChem.EmbedMolecule(mol_h, params)
            if embed_result == 0:
                AllChem.MMFFOptimizeMolecule(mol_h)
            else:
                log_debug(f"[Row {idx+1}] 3D embed failed for '{smiles}' -> 3D desc may be None.")

            # Compute 2D and 3D descriptors
            rd_descriptors = compute_2d_descriptors_rdkit(mol_h)
            rd3_descriptors = compute_3d_descriptors_rdkit(mol_h)
            rd_all = {**rd_descriptors, **rd3_descriptors}

            # Compute PubChem descriptors if enabled
            if use_pubchem:
                smi = Chem.MolToSmiles(mol_h, isomericSmiles=True)
                pc_descriptors = compute_descriptors_pubchem(smi)
                final_desc = merge_rdkit_and_pubchem_descriptors(rd_all, pc_descriptors)
            else:
                final_desc = merge_rdkit_and_pubchem_descriptors(rd_all, {})
        except Exception as e:
            log_debug(f"[Row {idx+1}] Skipping molecule due to descriptor error: {e}")
            continue

        # Extract BRIC fragments
        bric_fragments = []
        bric_id_list = set()
        try:
            fragments = BRICS.BreakBRICSBonds(mol)
            frags = Chem.GetMolFrags(fragments, asMols=True)
            used_brics = set()
            for bric_m in frags:
                bric_sm = Chem.MolToSmiles(bric_m)
                if bric_sm not in used_brics:
                    used_brics.add(bric_sm)
                    if calc_frag_props:
                        try:
                            # Full property calculation for the fragment
                            bric_info = bric_store.add_fragment(bric_sm, is_bbb_plus=is_bbb_plus, remove_brics_placeholder=True)
                            # Build the JSON entry copying all keys except SMILES and usage counts
                            bric_entry = {"BRIC": bric_sm, "ID": bric_info["ID"]}
                            for k, v in bric_info.items():
                                if k not in ("SMILES", "Total_count", "BBB+_count", "BBB-_count"):
                                    bric_entry[k] = v
                        except Exception as e:
                            log_debug(f"[Row {idx+1}] Skipping BRIC '{bric_sm}' => error: {e}")
                            bric_entry = {"BRIC": bric_sm, "ID": ""}
                    else:
                        # Minimal update: update usage counts in the store without including them in JSON
                        if bric_sm not in bric_store.data:
                            frag_id = f"{bric_store.prefix}{bric_store.counter:04d}"
                            bric_store.data[bric_sm] = {
                                "ID": frag_id,
                                "SMILES": bric_sm,
                                "Total_count": 0,
                                "BBB+_count": 0,
                                "BBB-_count": 0
                            }
                            bric_store.counter += 1
                        if is_bbb_plus:
                            bric_store.data[bric_sm]["BBB+_count"] += 1
                        else:
                            bric_store.data[bric_sm]["BBB-_count"] += 1
                        bric_store.data[bric_sm]["Total_count"] += 1
                        bric_info = bric_store.data[bric_sm]
                        # Build JSON entry with only SMILES and ID
                        bric_entry = {"BRIC": bric_sm, "ID": bric_info["ID"]}
                    bric_fragments.append(bric_entry)
                    bric_id_list.add(bric_entry["ID"])
        except Exception as e:
            bric_fragments = []
            log_debug(f"[Row {idx+1}] Error in BRICS: {e}")

        # Ring fragments
        ring_entries = []
        ring_id_list = set()
        try:
            ring_list, _ = process_ring_side_chains(smiles)  # returns (ring_list, side_list)
            used_rings = set()
            for ring_sm in ring_list:
                if ring_sm not in used_rings:
                    used_rings.add(ring_sm)
                    if calc_frag_props:
                        try:
                            ring_info = ring_store.add_fragment(ring_sm, is_bbb_plus=is_bbb_plus, remove_brics_placeholder=True)
                            ring_entry = {"RING": ring_sm, "ID": ring_info["ID"]}
                            for k, v in ring_info.items():
                                if k not in ("SMILES", "Total_count", "BBB+_count", "BBB-_count"):
                                    ring_entry[k] = v
                        except Exception as e:
                            log_debug(f"[Row {idx+1}] Skipping RING '{ring_sm}' => error: {e}")
                            ring_entry = {"RING": ring_sm, "ID": ""}
                    else:
                        if ring_sm not in ring_store.data:
                            frag_id = f"{ring_store.prefix}{ring_store.counter:04d}"
                            ring_store.data[ring_sm] = {
                                "ID": frag_id,
                                "SMILES": ring_sm,
                                "Total_count": 0,
                                "BBB+_count": 0,
                                "BBB-_count": 0
                            }
                            ring_store.counter += 1
                        if is_bbb_plus:
                            ring_store.data[ring_sm]["BBB+_count"] += 1
                        else:
                            ring_store.data[ring_sm]["BBB-_count"] += 1
                        ring_store.data[ring_sm]["Total_count"] += 1
                        ring_info = ring_store.data[ring_sm]
                        ring_entry = {"RING": ring_sm, "ID": ring_info["ID"]}
                    ring_entries.append(ring_entry)
                    ring_id_list.add(ring_entry["ID"])
        except Exception as e:
            ring_entries = []
            log_debug(f"[Row {idx+1}] Error extracting rings: {e}")

        # Side-chain fragments
        side_chain_entries = []
        sidechain_id_list = set()
        try:
            _, side_list = process_ring_side_chains(smiles)
            used_sidechains = set()
            for sc_sm in side_list:
                if sc_sm not in used_sidechains:
                    used_sidechains.add(sc_sm)
                    if calc_frag_props:
                        try:
                            sc_info = sidechain_store.add_fragment(sc_sm, is_bbb_plus=is_bbb_plus, remove_brics_placeholder=True)
                            sc_entry = {"SIDE_CHAIN": sc_sm, "ID": sc_info["ID"]}
                            for k, v in sc_info.items():
                                if k not in ("SMILES", "Total_count", "BBB+_count", "BBB-_count"):
                                    sc_entry[k] = v
                        except Exception as e:
                            log_debug(f"[Row {idx+1}] Skipping SIDE_CHAIN '{sc_sm}' => error: {e}")
                            sc_entry = {"SIDE_CHAIN": sc_sm, "ID": ""}
                    else:
                        if sc_sm not in sidechain_store.data:
                            frag_id = f"{sidechain_store.prefix}{sidechain_store.counter:04d}"
                            sidechain_store.data[sc_sm] = {
                                "ID": frag_id,
                                "SMILES": sc_sm,
                                "Total_count": 0,
                                "BBB+_count": 0,
                                "BBB-_count": 0
                            }
                            sidechain_store.counter += 1
                        if is_bbb_plus:
                            sidechain_store.data[sc_sm]["BBB+_count"] += 1
                        else:
                            sidechain_store.data[sc_sm]["BBB-_count"] += 1
                        sidechain_store.data[sc_sm]["Total_count"] += 1
                        sc_info = sidechain_store.data[sc_sm]
                        sc_entry = {"SIDE_CHAIN": sc_sm, "ID": sc_info["ID"]}
                    side_chain_entries.append(sc_entry)
                    sidechain_id_list.add(sc_entry["ID"])
        except Exception as e:
            side_chain_entries = []
            log_debug(f"[Row {idx+1}] Error extracting side chains: {e}")

        # Build the JSON entry for the molecule
        mol_entry = {
            "NO.": no_val,
            "SMILES": smiles,
            "BBB+/BBB-": bbb_status,
            "group": grp,
            "BRICs": bric_fragments,
            "RINGS": ring_entries,
            "SIDE_CHAINS": side_chain_entries,
            "BRIC_IDs": "; ".join(sorted(list(bric_id_list))),
            "RING_IDs": "; ".join(sorted(list(ring_id_list))),
            "SIDECHAIN_IDs": "; ".join(sorted(list(sidechain_id_list)))
        }
        # Add molecule-level descriptors
        for k, v in final_desc.items():
            mol_entry[k] = v

        json_data.append(mol_entry)
        molecule_dicts.append(mol_entry)

    # Finalize usage counts in fragment stores
    bric_store.finalize_counts()
    ring_store.finalize_counts()
    sidechain_store.finalize_counts()

    # Save JSON output
    base, ext = os.path.splitext(output_json)
    try:
        with open(output_json, 'w') as jf:
            json.dump(json_data, jf, indent=4)
        print(f"JSON data saved to '{output_json}'.")

        min_json = f"{base}.min{ext}"
        with open(min_json, 'w') as mf:
            json.dump(json_data, mf, separators=(',', ':'))
        print(f"Minified JSON data saved to '{min_json}'.")
    except Exception as e:
        print(f"Error writing JSON: {e}")
        return

    # Generate molecule-level CSV excluding large list columns
    df_molecules = pd.DataFrame(molecule_dicts)
    for col_to_drop in ("BRICs", "RINGS", "SIDE_CHAINS"):
        if col_to_drop in df_molecules.columns:
            df_molecules.drop(columns=[col_to_drop], inplace=True)

    molecules_csv = f"{base}_molecules.csv"
    df_molecules.to_csv(molecules_csv, index=False)
    print(f"Molecule-level CSV saved to '{molecules_csv}'.")

    # Save fragment-level CSV files
    df_brics = bric_store.all_rows_df()
    brics_csv = f"{base}_brics.csv"
    df_brics.to_csv(brics_csv, index=False)
    print(f"BRIC-level CSV saved to '{brics_csv}'.")

    df_rings = ring_store.all_rows_df()
    rings_csv = f"{base}_rings.csv"
    df_rings.to_csv(rings_csv, index=False)
    print(f"Ring-level CSV saved to '{rings_csv}'.")

    df_sidechains = sidechain_store.all_rows_df()
    sidechains_csv = f"{base}_sidechains.csv"
    df_sidechains.to_csv(sidechains_csv, index=False)
    print(f"Side-chain-level CSV saved to '{sidechains_csv}'.")

    # Generate summary statistics
    summary_frames = []
    if not df_molecules.empty:
        summary_frames.append(compute_summary_stats(df_molecules, "molecule"))
    if not df_brics.empty:
        summary_frames.append(compute_summary_stats(df_brics, "BRIC"))
    if not df_rings.empty:
        summary_frames.append(compute_summary_stats(df_rings, "RING"))
    if not df_sidechains.empty:
        summary_frames.append(compute_summary_stats(df_sidechains, "SIDE_CHAIN"))
    if summary_frames:
        combined_summary = pd.concat(summary_frames, ignore_index=True)
    else:
        combined_summary = pd.DataFrame(columns=["Type", "Metric", "Mean", "SD"])

    summary_csv = f"{base}_summary.csv"
    combined_summary.to_csv(summary_csv, index=False)
    print(f"Data-similarity summary CSV saved to '{summary_csv}'.")

    # Missing data check
    error_log = []
    property_wise_log = []

    def _check(df_in, tname, id_col, smiles_col=None):
        if not df_in.empty:
            check_for_missing_data(df_in, tname, id_col, error_log, property_wise_log, smiles_col)

    _check(df_molecules, "molecule", "NO.", "SMILES")
    _check(df_brics, "BRIC", "ID", "SMILES")
    _check(df_rings, "RING", "ID", "SMILES")
    _check(df_sidechains, "SIDE_CHAIN", "ID", "SMILES")

    object_summary = defaultdict(set)
    for line in error_log:
        if line.startswith("[") and "] " in line:
            bracket_end = line.index("]")
            the_type = line[1:bracket_end]
            splitted = line.split("in ")
            if len(splitted) >= 2:
                trailing = splitted[-1]
                eq_split = trailing.split("=")
                if len(eq_split) >= 2:
                    row_id = eq_split[-1].strip().strip("'")
                    object_summary[the_type].add(row_id)

    summary_lines = []
    summary_lines.append("Missing Data Summary:\n")
    for ttype in ["molecule", "BRIC", "RING", "SIDE_CHAIN"]:
        count_missing_objs = len(object_summary[ttype])
        summary_lines.append(f"  {ttype}: {count_missing_objs} with missing data\n")

    summary_lines.append("\nDetailed Missing Data Lines:\n")
    for e in error_log:
        summary_lines.append(e)

    summary_lines.append("\nProperty-wise Missing Data:\n")
    summary_lines.append("type,SMILES_or_ID,Descriptor\n")
    for pw in property_wise_log:
        row_str = f"{pw['Type']},{pw['SMILES']},{pw['Descriptor']}"
        summary_lines.append(row_str)

    # Save error summary
    error_file = f"{base}_error_summary.txt"
    with open(error_file, "w") as ef:
        ef.write("\n".join(summary_lines))
    print(f"Error summary log saved to '{error_file}'.")
    print("Done. All processing complete.")


if __name__ == "__main__":
    """
    Command-line interface for generating JSON from CSV with molecular descriptors.

    - Parses command-line arguments for input CSV, output JSON, and PubChem usage.
    - If arguments are missing, prompts the user for manual input.
    - Ensures the output directory exists before processing.
    - Calls 'generate_json_from_csv()' with the provided parameters.
    """
    # Initialize argument parser
    parser = argparse.ArgumentParser(description="Generate JSON from CSV with descriptors.")
    
    # Define command-line arguments
    parser.add_argument("--input_csv", help="Path to input CSV.")
    parser.add_argument("--output_json", help="Path to output JSON.")
    parser.add_argument("--use_pubchem", choices=["y", "n"], default="n", help="Use PubChem y/n.")
    parser.add_argument(
        "--calculate_fragment_properties",
        choices=["y", "n"],
        default="n",
        help="Calculate fragment properties? (y/n, default: n)"
    )
    
    # Parse command-line arguments
    args = parser.parse_args()
    
    # Check if required arguments are provided via command line
    if args.input_csv and args.output_json:
        use_pubchem = args.use_pubchem == "y"
        calc_frag_props = args.calculate_fragment_properties == "y"
        generate_json_from_csv(args.input_csv, args.output_json, use_pubchem, calc_frag_props)
    
    # Otherwise, prompt the user for input with defaults.
    else:
        input_csv = input("Enter input CSV path (or press Enter for default 'data/B3DB_full.csv'): ")
        if not input_csv.strip():
            input_csv = "data/B3DB_full.csv"

        output_json = input("Enter output JSON path (or press Enter for default 'data/B3DB_processed/processed.json'): ")
        if not output_json.strip():
            output_json = "data/B3DB_processed/processed.json"
        
        # Ensure the output directory exists
        output_dir = os.path.dirname(output_json)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            print(f"Created directories: {output_dir}")

        use_pubchem = input("Search PubChem? (y/n, default n.): ").strip().lower()
        if not use_pubchem:
            use_pubchem = "n"
        use_pubchem = use_pubchem == "y"
        
        calc_frag_props = input("Calculate fragment properties? (y/n, default n.): ").strip().lower()
        if not calc_frag_props:
            calc_frag_props = "n"
        calc_frag_props = calc_frag_props == "y"
        
        generate_json_from_csv(input_csv, output_json, use_pubchem, calc_frag_props)