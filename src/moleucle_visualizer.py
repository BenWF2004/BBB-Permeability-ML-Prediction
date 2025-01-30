"""
Author: Ben Franey
Version: 1.0.2
Date: 23-01-2025
Overview:
    This script converts chemical identifiers (SMILES, IUPAC, or General names) to their corresponding SMILES
    representations and generates SVG images of the molecules. It can accept inputs via command-line arguments
    or interactively through user prompts.
    Primarity used as part of web interface
"""

import sys
import os
import json
import argparse
from rdkit import Chem
from rdkit.Chem import Draw, AllChem
import pubchempy

def is_valid_smiles(smiles):
    """
    Validate a given SMILES string.

    Overview:
        - Checks if the input SMILES string is valid using RDKit.
        - If invalid, attempts to retrieve the correct canonical SMILES from PubChem.
        - Returns a valid SMILES string if found; otherwise, returns None.

    Parameters:
        smiles (str): The input SMILES string.

    Returns:
        str | None: 
            - The validated SMILES string if valid.
            - Canonical SMILES from PubChem if correction is available.
            - None if no valid structure is found.

    Raises:
        None (Handles exceptions internally).
    """
    
    try:
        # Check validity using RDKit
        molecule = Chem.MolFromSmiles(smiles)
        if molecule:
            return smiles
    except Exception:
        pass

    # If invalid, search PubChem
    try:
        compounds = pubchempy.get_compounds(smiles, namespace='smiles')
        if compounds:
            return compounds[0].canonical_smiles
    except Exception:
        pass

    # Return None if no valid SMILES is found
    return None

def iupac_to_smiles(iupac_name):
    """
    Convert an IUPAC name to a SMILES string using PubChem.

    Overview:
        - Uses PubChemPy to retrieve the SMILES representation of a given IUPAC name.
        - Returns the isomeric SMILES if available; otherwise, the canonical SMILES.

    Parameters:
        iupac_name (str): The IUPAC chemical name.

    Returns:
        str | None: 
            - The isomeric or canonical SMILES string if found.
            - None if the compound is not available in PubChem.

    Raises:
        None (Handles exceptions internally).
    """
    
    try:
        compounds = pubchempy.get_compounds(iupac_name, 'name')
        if compounds:
            compound = compounds[0]
            return compound.isomeric_smiles if compound.isomeric_smiles else compound.canonical_smiles
    except Exception:
        pass
    return None

def save_to_svg(molecule, output_file):
    try:
        non_hydrogen_count = sum(1 for atom in molecule.GetAtoms() if atom.GetSymbol() != 'H')
        square_size = int(150 * (non_hydrogen_count ** 0.45))
        Draw.MolToFile(molecule, output_file, format='svg', size=(square_size, square_size))
    except Exception as e:
        raise RuntimeError(f"Error saving SVG: {str(e)}")

def main(input_type, identifier, output_folder):
    """
    Save an RDKit molecule structure as an SVG image.

    Overview:
        - Determines the image size dynamically based on the number of non-hydrogen atoms.
        - Uses RDKit's `MolToFile` function to save the molecule as an SVG.

    Parameters:
        molecule (Chem.Mol): The RDKit molecule object.
        output_file (str): Path to the output SVG file.

    Returns:
        None (Saves the SVG file).

    Raises:
        RuntimeError: If an error occurs during SVG generation.
    """
    
    os.makedirs(output_folder, exist_ok=True)

    molecule = None
    smiles = None

    try:
        if input_type.upper() == "SMILES":
            smiles = is_valid_smiles(identifier)
            if not smiles:
                raise ValueError("Invalid SMILES")
            molecule = Chem.MolFromSmiles(smiles)

        elif input_type.upper() == "IUPAC":
            smiles = iupac_to_smiles(identifier)
            if not smiles:
                raise ValueError("Invalid IUPAC Name")
            smiles = is_valid_smiles(smiles)
            if not smiles:
                raise ValueError("Invalid SMILES derived from IUPAC Name")
            molecule = Chem.MolFromSmiles(smiles)

        elif input_type.upper() == "GENERAL":
            try:
                compound = pubchempy.get_compounds(identifier, 'name')[0]
                smiles = compound.isomeric_smiles if compound.isomeric_smiles else compound.canonical_smiles
            except IndexError:
                raise ValueError("Invalid General Name")
            smiles = is_valid_smiles(smiles)
            if not smiles:
                raise ValueError("Invalid SMILES derived from General Name")
            molecule = Chem.MolFromSmiles(smiles)
        else:
            raise ValueError("Invalid input type. Must be SMILES, IUPAC, or GENERAL.")

        sanitized_name = ''.join(c if c.isalnum() else '_' for c in identifier)
        svg_file = os.path.join(output_folder, f"{sanitized_name}.svg")

        AllChem.Compute2DCoords(molecule)
        save_to_svg(molecule, svg_file)

        # Return JSON output
        print(json.dumps({
            "success": True,
            "smiles": smiles,
            "svgPath": svg_file
        }))

    except Exception as e:
        print(json.dumps({"success": False, "error": str(e)}))
        sys.exit(1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert chemical identifiers to SMILES and generate SVG images.")
    parser.add_argument('--type', type=str, choices=['SMILES', 'IUPAC', 'GENERAL'], help='Type of the identifier (SMILES, IUPAC, GENERAL)')
    parser.add_argument('--identifier', type=str, help='The chemical identifier (SMILES string, IUPAC name, or general name)')
    parser.add_argument('--output_folder', type=str, help='Folder to save the SVG output')

    args = parser.parse_args()

    # Check if all arguments are provided
    if args.type and args.identifier and args.output_folder:
        input_type = args.type
        identifier = args.identifier
        output_folder = args.output_folder
    else:
        print("Some or all command-line arguments are missing. Switching to interactive mode.")
        if not args.type:
            input_type = input("Enter the type of identifier (SMILES, IUPAC, GENERAL): ").strip()
            while input_type.upper() not in ['SMILES', 'IUPAC', 'GENERAL']:
                print("Invalid type. Please enter 'SMILES', 'IUPAC', or 'GENERAL'.")
                input_type = input("Enter the type of identifier (SMILES, IUPAC, GENERAL): ").strip()
        else:
            input_type = args.type

        if not args.identifier:
            identifier = input("Enter the chemical identifier: ").strip()
        else:
            identifier = args.identifier

        if not args.output_folder:
            output_folder = input("Enter the output folder path: ").strip()
        else:
            output_folder = args.output_folder

    main(input_type, identifier, output_folder)
