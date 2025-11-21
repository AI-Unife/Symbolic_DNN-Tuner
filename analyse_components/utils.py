# In utils.py
"""
This module provides miscellaneous utility functions used across the analysis.

Includes helpers for:
- Parsing metadata (Dataset, Tuner, etc.) from experiment folder names.
- Heuristically extracting a 'tuner' name as a fallback.
- Copying and selecting representative log files for inspection.
- Filtering experiment directories based on a name prefix.
"""

import shutil
import logging
from pathlib import Path
from typing import Optional, Dict, Any

def extract_tuner(exp_name: str) -> Optional[str]:
    """
    Heuristically infers the tuner name from the experiment folder string.
    
    This is used as a FALLBACK if the 10-part name parsing fails.

    Args:
        exp_name: The name of the experiment directory.

    Returns:
        A string (e.g., "RS_ruled", "filtered") or None if not found.
    """
    tuner_names = ["filtered", "basic", "standard", "RS_ruled", "RS"]
    for t in tuner_names:
        if t in exp_name: 
            return t
            
    # Fallback heuristic based on string splitting
    parts = exp_name.split("_")
    if len(parts) > 7:
        cand = parts[6]
        if parts[7] == "ruled": 
            cand += "_ruled"
        return cand
        
    logging.debug("Could not determine tuner name from: %s", exp_name)
    return None

def parse_experiment_name(exp_name: str) -> Dict[str, Any]:
    """
    Parses an experiment name with the 10-part format:
    AAAA_MM_GG_HH_CODE_DATASET_OPT_SEED_EVAL_EPOCHS
    
    Returns a dictionary with parsed values.
    Falls back to 'extract_tuner' if format doesn't match.
    """
    parts = exp_name.split('_')
    parsed_data = {
        'Dataset': None,
        'Tuner': None, # 'OPT'
        'Seed': None,
        'Eval': None,
        'Epochs': None
    }
    
    if len(parts) >= 11:
        try:
            parsed_data['Dataset'] = parts[5]
            if parts[7] == 'ruled':
                parsed_data['Tuner'] = parts[6] + '_' + parts[7]     # This is 'OPT'
                parsed_data['Seed'] = int(parts[8])
                parsed_data['Eval'] = int(parts[9])
                parsed_data['Epochs'] = int(parts[10])
            else:
                parsed_data['Tuner'] = parts[6]     # This is 'OPT'
                parsed_data['Seed'] = int(parts[7])
                parsed_data['Eval'] = int(parts[8])
                parsed_data['Epochs'] = int(parts[9])
            logging.debug("Parsed folder name '%s' successfully.", exp_name)
        except Exception as e:
            logging.error("Failed to parse 11-part experiment name '%s': %s. Falling back.", exp_name, e)
            # Fallback for Tuner if parsing fails
            parsed_data['Tuner'] = extract_tuner(exp_name)
    else:
        logging.warning("Experiment name '%s' does not match 10-part format. Falling back to 'extract_tuner'.", exp_name)
        # Fallback for Tuner
        parsed_data['Tuner'] = extract_tuner(exp_name)
        
    return parsed_data


def copy_log_to_output(log_src: Path, exp_dir: Path) -> Optional[Path]:
    """
    Copies a representative log to <exp_dir>/output/output.log for inspection.
    This provides a stable, easy-to-find log file for manual review.

    Args:
        log_src: The source log file to copy.
        exp_dir: The root directory of the experiment.

    Returns:
        The Path to the destination file, or None on failure.
    """
    dest_dir = exp_dir / "output"
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / "output.log"
    try:
        shutil.copy2(log_src, dest)
        logging.info("Copied log: %s -> %s", log_src.name, dest)
        return dest
    except Exception as e:
        logging.warning("Failed to copy %s to %s: %s", log_src, dest, e)
        return None

def pick_representative_log(exp_dir: Path) -> Optional[Path]:
    """
    Chooses a log to copy: 
    1.  Prefers 'output.log' if it already exists.
    2.  Falls back to the largest .out file found.

    Args:
        exp_dir: The root directory of the experiment.

    Returns:
        A Path object to the chosen log file, or None if none are found.
    """
    # 1. Check for the explicitly named file
    explicit = exp_dir / "output.log"
    if explicit.is_file(): 
        return explicit
        
    # 2. Find all .out files and pick the largest
    outs = list(exp_dir.rglob("*.out"))
    if not outs: 
        return None
        
    # Pick the largest by size as it's most likely the main log
    try:
        outs.sort(key=lambda p: p.stat().st_size if p.exists() else 0, reverse=True)
    except FileNotFoundError:
        logging.warning("File not found during log sorting, skipping.")
        return None
        
    return outs[0]

def should_select_experiment_dir(dir_name: str, prefix: str) -> bool:
    """
    Return True if the folder should be processed.
    
    Args:
        dir_name: The name of the experiment directory.
        prefix: The user-supplied prefix (can be empty).

    Returns:
        True if the directory name starts with the prefix.
    """
    # An empty prefix should match everything
    return dir_name.startswith(prefix)