"""
Author: Ben Franey
Version: 1.2 - Publish: 1.0
Last Review Date: 30-01-2025
Overview:
Test script only.
To create example BRICS, RING and SIDE CHAIN decomposition arrays and SVGs.
"""


import os
import argparse
from rdkit import Chem
from rdkit.Chem import BRICS, Draw
from rdkit.Chem.rdchem import Mol
from typing import List, Tuple


def log_debug(message: str, output_dir: str) -> None:
    """
    Logs debug messages to the terminal and a file.

    Parameters:
        message (str): The message to log.
        output_dir (str): Directory to store the log file.

    Writes log messages to "decomposition.txt" inside output_dir.
    """
    print(message)
    log_file_path = os.path.join(output_dir, "decomposition.txt")
    with open(log_file_path, "a") as log_file:
        log_file.write(message + "\n")


def process_ring_side_chains(smiles: str) -> Tuple[List[str], List[str]]:
    """
    Extracts ring fragments and side-chain fragments from a molecule SMILES.

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
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return rings_list, side_chains_list

        ring_info = mol.GetRingInfo()
        rings = ring_info.AtomRings()

        for ring in rings:
            ring_sm = Chem.MolFragmentToSmiles(mol, atomsToUse=ring)
            rings_list.append(ring_sm)

        ring_atoms = set(a for ring in rings for a in ring)
        non_ring_atoms = [
            atom.GetIdx()
            for atom in mol.GetAtoms()
            if atom.GetIdx() not in ring_atoms
        ]

        visited_atoms = set()
        for atom_idx in non_ring_atoms:
            if atom_idx not in visited_atoms:
                neighbors = [atom_idx] + [
                    nbr.GetIdx()
                    for nbr in mol.GetAtomWithIdx(atom_idx).GetNeighbors()
                    if nbr.GetIdx() not in ring_atoms and nbr.GetIdx() not in visited_atoms
                ]

                fragment = Chem.MolFragmentToSmiles(mol, atomsToUse=neighbors)
                side_chains_list.append(fragment)
                visited_atoms.update(neighbors)

    except Exception:
        return rings_list, side_chains_list

    return rings_list, side_chains_list


def process_brics(smiles: str) -> List[str]:
    """
    Performs BRICS decomposition on a molecule.

    Parameters:
        smiles (str): SMILES representation of the molecule.

    Returns:
        list: SMILES strings of BRICS fragments.
    """
    fragments = []
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return fragments

        brics_fragments = BRICS.BRICSDecompose(mol)
        fragments = list(brics_fragments)

    except Exception:
        return fragments

    return fragments


def save_svg(mols: List[Mol], folder: str, prefix: str) -> None:
    """
    Saves SVG images of molecular fragments.

    Parameters:
        mols (list): List of RDKit molecule objects.
        folder (str): Destination folder for images.
        prefix (str): Prefix for filenames.
    """
    if not os.path.exists(folder):
        os.makedirs(folder)

    for i, mol in enumerate(mols):
        if mol:
            filename = os.path.join(folder, f"{prefix}_fragment_{i + 1}.svg")
            Draw.MolToFile(mol, filename, format="svg")


def save_whole_molecule_svg(smiles: str, folder: str) -> None:
    """
    Saves an SVG image of the entire molecule.

    Parameters:
        smiles (str): SMILES representation of the molecule.
        folder (str): Directory where the SVG will be stored.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol:
        filename = os.path.join(folder, "whole_molecule.svg")
        Draw.MolToFile(mol, filename, format="svg")


def main():
    """
    Main execution function.

    - Accepts user input via --smiles and --output_dir.
    - Falls back to input() if arguments are not provided.
    - Runs BRICS, ring, and side-chain decomposition.
    - Logs results in a user-specified output directory.
    - Saves the whole molecule and fragment SVGs in the user-defined output folder.
    """
    parser = argparse.ArgumentParser(description="Decompose a SMILES string into BRICS, rings, and side chains.")
    parser.add_argument("--smiles", type=str, help="SMILES string of the molecule")
    parser.add_argument("--output_dir", type=str, help="Directory to store output files")
    args = parser.parse_args()

    # Fallback to input() if no command-line arguments are provided
    smiles = args.smiles if args.smiles else input("Enter SMILES string: ")
    output_dir = args.output_dir if args.output_dir else input("Enter output directory: ")

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    log_debug(f"Processing SMILES: {smiles}", output_dir)

    # Save whole molecule SVG
    save_whole_molecule_svg(smiles, output_dir)

    # BRICS decomposition
    brics_fragments = process_brics(smiles)
    log_debug(f"BRICS Fragments ({len(brics_fragments)}): {brics_fragments}", output_dir)

    # Rings & Side-Chains decomposition
    rings, side_chains = process_ring_side_chains(smiles)
    log_debug(f"Ring Fragments ({len(rings)}): {rings}", output_dir)
    log_debug(f"Side-Chain Fragments ({len(side_chains)}): {side_chains}", output_dir)

    # Convert fragments to RDKit Mol objects
    brics_mols = [Chem.MolFromSmiles(frag) for frag in brics_fragments]
    rings_mols = [Chem.MolFromSmiles(frag) for frag in rings]
    side_chain_mols = [Chem.MolFromSmiles(frag) for frag in side_chains]

    # Save SVGs
    save_svg(brics_mols, output_dir, "brics")
    save_svg(rings_mols, output_dir, "rings")
    save_svg(side_chain_mols, output_dir, "sidechains")

    log_debug(f"All outputs saved in {output_dir}", output_dir)


if __name__ == "__main__":
    main()
