"""
Author: Ben Franey
Version: 11.1.6 - Publish: 1.0
Last Review Date: 30-01-2025
Overview:
Test script only.
To cut Json produced and remove BRIC/RING/SIDECHAIN information.
"""

import os
import json
import sys

def process_json_file(filepath):
    """
    Process a single JSON file to filter specific properties from molecular fragment sections.

    Overview:
        - Removes all properties from 'BRICs', 'RINGS', and 'SIDE_CHAINS' sections except:
          - 'BRIC' and 'ID' for BRICs
          - 'RING' and 'ID' for RINGS
          - 'SIDE_CHAIN' and 'ID' for SIDE_CHAINS
        - Saves two output JSON files:
          1. A formatted (pretty-printed) JSON file with the suffix `.cut.json`
          2. A minimized (compact) JSON file with the suffix `.cut.min.json`

    Parameters:
        filepath (str): Path to the input JSON file.

    Returns:
        None (Outputs two JSON files).

    Raises:
        - JSONDecodeError: If the input file contains invalid JSON.
        - FileNotFoundError: If the file does not exist.
        - IOError: If there is an issue reading/writing the file.
    """
    try:
        with open(filepath, 'r', encoding='utf-8') as file:
            data = json.load(file)
    except json.JSONDecodeError as e:
        print(f"Failed to decode JSON from {filepath}: {e}")
        return
    except Exception as e:
        print(f"Error reading file {filepath}: {e}")
        return

    # Define sections to process and their respective keys to keep
    sections = {
        'BRICs': ['BRIC', 'ID'],
        'RINGS': ['RING', 'ID'],
        'SIDE_CHAINS': ['SIDE_CHAIN', 'ID']
    }

    for item in data:
        for section, keys_to_keep in sections.items():
            if section in item:
                new_section = []
                for entry in item[section]:
                    # Keep only the specified keys if they exist in the entry
                    filtered_entry = {key: entry[key] for key in keys_to_keep if key in entry}
                    new_section.append(filtered_entry)
                item[section] = new_section

    # Prepare output filenames
    dir_name, base_name = os.path.split(filepath)
    name, ext = os.path.splitext(base_name)
    cut_json_path = os.path.join(dir_name, f"{name}.cut.json")
    cut_min_json_path = os.path.join(dir_name, f"{name}.cut.min.json")

    # Save the formatted (pretty-printed) JSON
    try:
        with open(cut_json_path, 'w', encoding='utf-8') as file:
            json.dump(data, file, indent=4)
        print(f"Saved condensed JSON to {cut_json_path}")
    except Exception as e:
        print(f"Error writing to file {cut_json_path}: {e}")

    # Save the minimized JSON
    try:
        with open(cut_min_json_path, 'w', encoding='utf-8') as file:
            json.dump(data, file, separators=(',', ':'))
        print(f"Saved minimized JSON to {cut_min_json_path}")
    except Exception as e:
        print(f"Error writing to file {cut_min_json_path}: {e}")

if __name__ == "__main__":
    filepath = input("Enter directory of JSON to cut: ")
    process_json_file(filepath)
