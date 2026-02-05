#!/usr/bin/env python3
"""
Script to expand YAML event config files when concepts are stored as directories.

Given an input directory and a YAML config file, this script:
1. Checks if concept names (like "notes", "procedure") are directories
2. If they are directories, expands the YAML to include all files in those directories
3. For example, if "notes" is a directory with chunk_0.csv, chunk_1.csv, it creates
   entries for "notes/chunk_0", "notes/chunk_1", etc.
"""

import argparse
import copy
import os
import yaml
from pathlib import Path
from typing import Dict, Any


def get_files_in_directory(dir_path: Path, extensions: tuple = ('.csv', '.parquet')) -> list[str]:
    """Get all files in a directory with specified extensions, sorted by name.
    
    Args:
        dir_path: Path to the directory
        extensions: Tuple of file extensions to include
        
    Returns:
        List of file names (without extension) sorted alphabetically
    """
    if not dir_path.exists() or not dir_path.is_dir():
        return []
    
    files = []
    for file_path in sorted(dir_path.iterdir()):
        if file_path.is_file() and file_path.suffix in extensions:
            # Return filename without extension
            files.append(file_path.stem)
    return files


def expand_config_for_directories(
    config: Dict[str, Any], 
    input_dir: Path,
    extensions: tuple = ('.csv', '.parquet')
) -> Dict[str, Any]:
    """Expand YAML config to include all files in directories.
    
    Args:
        config: The loaded YAML config dictionary
        input_dir: Path to the input directory containing the data files
        extensions: File extensions to look for
        
    Returns:
        Expanded config dictionary
    """
    # Keys to skip (these are special config keys, not concept names)
    skip_keys = {'subject', 'subject_id_col'}
    
    expanded_config = {}
    
    # Copy special keys first
    if 'subject_id_col' in config:
        expanded_config['subject_id_col'] = config['subject_id_col']
    if 'subject' in config:
        expanded_config['subject'] = config['subject']
    
    # Process each concept in the config
    for concept_name, concept_config in config.items():
        if concept_name in skip_keys:
            continue
        
        concept_path = input_dir / concept_name
        
        # Check if this concept exists as a directory
        if concept_path.exists() and concept_path.is_dir():
            # Get all files in the directory
            files = get_files_in_directory(concept_path, extensions)
            
            if files:
                # Expand: create entries for each file
                # Copy the entire concept_config structure to each file entry
                for file_name in files:
                    # Create entry like "notes/chunk_0"
                    expanded_key = f"{concept_name}/{file_name}"
                    # Copy all configs under the concept (deep copy to avoid reference issues)
                    expanded_config[expanded_key] = copy.deepcopy(concept_config)
                
                print(f"Expanded '{concept_name}' directory: found {len(files)} files")
            else:
                print(f"Warning: '{concept_name}' is a directory but contains no {extensions} files")
                # Keep original entry if directory is empty
                expanded_config[concept_name] = concept_config
        else:
            # Not a directory, keep as-is (might be a file or doesn't exist yet)
            expanded_config[concept_name] = concept_config
    
    return expanded_config


def main():
    parser = argparse.ArgumentParser(
        description="Expand YAML event config to include all files in directories"
    )
    parser.add_argument(
        "input_dir",
        type=str,
        help="Input directory containing the data files/directories"
    )
    parser.add_argument(
        "config_file",
        type=str,
        help="Path to the input YAML config file"
    )
    parser.add_argument(
        "-o", "--output",
        type=str,
        help="Path to output YAML file (default: overwrites input file)"
    )
    parser.add_argument(
        "--extensions",
        nargs="+",
        default=[".csv", ".parquet"],
        help="File extensions to include when expanding directories (default: .csv .parquet)"
    )
    
    args = parser.parse_args()
    
    input_dir = Path(args.input_dir)
    config_path = Path(args.config_file)
    output_path = Path(args.output) if args.output else config_path
    extensions = tuple(args.extensions)
    
    # Validate inputs
    if not input_dir.exists():
        raise ValueError(f"Input directory does not exist: {input_dir}")
    if not input_dir.is_dir():
        raise ValueError(f"Input path is not a directory: {input_dir}")
    if not config_path.exists():
        raise ValueError(f"Config file does not exist: {config_path}")
    
    # Load YAML config
    print(f"Loading config from: {config_path}")
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    if not config:
        raise ValueError("Config file is empty or invalid")
    
    # Expand config for directories
    print(f"Checking for directories in: {input_dir}")
    expanded_config = expand_config_for_directories(config, input_dir, extensions)
    
    # Write expanded config
    print(f"Writing expanded config to: {output_path}")
    with open(output_path, 'w') as f:
        yaml.dump(expanded_config, f, default_flow_style=False, sort_keys=False)
    
    print("Done!")


if __name__ == "__main__":
    main()
