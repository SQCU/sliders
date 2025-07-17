"""
This script implements the first step of the 'Discovery Generator' as outlined in the
'd_dset_demolisher.py' critique. Its primary function is to scan the filesystem for image
files based on a provided configuration and organize them into a structured 'data pool'.

This module is designed to be both executable as a standalone script for self-testing
and importable as a reusable tool in other Python programs.

Usage:
  1. As a standalone script (for self-test/default behavior):
     `python discovery_scanner.py`
     This will load the 'experiment_manifest_xl.yaml' from the default path
     ('run_artifacts/yamlzoo/' relative to the project root) and print the discovered data.

  2. As an importable module (for programmatic use):
     `from discovery_scanner import call_from_path`
     Then, call `discovered_data = call_from_path(path_to_manifest_yaml)`

Expected Input for `call_from_path`:
  `manifest_path` (pathlib.Path): The absolute or relative path to a YAML configuration file.
  This YAML file must contain a 'data_setup.config' section specifying:
    - 'folder_main': The base directory for image datasets (relative to project root).
    - 'folders': A list of subfolder names, each corresponding to a specific 'scale'.
    - 'scales': A list of numerical scale values, matching the 'folders' list.

Output of `call_from_path`:
  `dict`: A dictionary representing the discovered data pool. Keys are image filenames
  (e.g., 'image.png') and values are dictionaries mapping scale values (float) to
  absolute image paths (str).
  Example: {'image.png': {0.15: '/path/to/0/image.png', 1.0: '/path/to/1/image.png'}}
  Returns an empty dictionary if the manifest file is not found, cannot be parsed,
  or if 'data_setup.config' is missing/invalid, or if no image data is discovered.

Scope:
  This script is responsible for the initial discovery phase of the data pipeline.
  It identifies all relevant image assets and their locations across different scales,
  producing a comprehensive 'data pool' that serves as input for subsequent stages
  like schedule generation and asset materialization. It handles filesystem I/O
  and basic data structuring for this discovery process.
"""

import yaml
from pathlib import Path
from collections import defaultdict
import os

def discover_data_pool(config: dict) -> dict:
    """
    Scans the filesystem for image paths based on the provided configuration.
    Groups images by filename across different scale folders.

    Args:
        config (dict): A dictionary containing data setup configuration, 
                       including 'folder_main', 'folders', and 'scales'.

    Returns:
        dict: A dictionary where keys are image filenames (e.g., 'image.png')
              and values are dictionaries mapping scale values (float) to 
              absolute image paths (str).
              Example: {'image.png': {0.15: '/path/to/0/image.png', 1.0: '/path/to/1/image.png'}}
    """
    data_pool = defaultdict(dict)
    
    # Ensure scales are floats for consistent lookup
    scales = [float(s) for s in config['scales']]

    # Construct the absolute path to the main data folder
    # Assuming the script is run from the project root, or adjust as needed
    # For this example, we'll assume the project root is the current working directory
    project_root = Path(os.getcwd())
    folder_main_abs = project_root / config['folder_main']

    if not folder_main_abs.is_dir():
        print(f"Error: Main data folder not found: {folder_main_abs}")
        return {}

    for folder_name, scale in zip(config['folders'], scales):
        subfolder_path = folder_main_abs / folder_name
        
        if not subfolder_path.is_dir():
            print(f"Warning: Subfolder not found, skipping: {subfolder_path}")
            continue

        for image_path in subfolder_path.glob("*"):
            if image_path.suffix.lower() in ['.png', '.jpg', '.jpeg', '.webp']:
                data_pool[image_path.name][scale] = str(image_path.resolve()) # Store absolute path
    
    return dict(data_pool)

def call_from_path(manifest_path: Path) -> dict:
    """
    Loads a YAML manifest from the given path, extracts the data setup configuration,
    and discovers the image data pool.

    Args:
        manifest_path (pathlib.Path): The path to the YAML manifest file.

    Returns:
        dict: The discovered data pool, or an empty dictionary if an error occurs.
    """
    if not manifest_path.exists():
        print(f"Error: Manifest file not found: {manifest_path}")
        return {}

    try:
        with open(manifest_path, 'r') as f:
            manifest_config = yaml.safe_load(f)
    except yaml.YAMLError as e:
        print(f"Error parsing YAML manifest {manifest_path}: {e}")
        return {}

    data_config = manifest_config.get('data_setup', {}).get('config', {})
    if not data_config:
        print("Error: 'data_setup.config' not found in the manifest.")
        return {}

    print(f"Discovering data pool using config from: {manifest_path}")
    discovered_data = discover_data_pool(data_config)
    return discovered_data

def main():
    # Default path for self-testing
    project_root = Path(os.getcwd())
    default_manifest_path = project_root / "run_artifacts" / "yamlzoo" / "experiment_manifest_xl.yaml"

    discovered_data = call_from_path(default_manifest_path)

    if discovered_data:
        print("--- Discovered Data Pool ---")
        for filename, scales_map in discovered_data.items():
            print(f"File: {filename}")
            for scale, path in scales_map.items():
                print(f"  Scale {scale}: {path}")
    else:
        print("No image data discovered or an error occurred.")

if __name__ == "__main__":
    main()
