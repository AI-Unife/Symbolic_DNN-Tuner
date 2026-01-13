# In parsing.py
"""
This module handles the parsing of raw experiment output files.

It defines regular expressions to extract metadata (like dataset, epochs)
from .out or .log files and provides functions to parse the core
'acc_report.txt' and 'hyper-neural.txt' files to extract the results
for every network evaluated.
"""

import yaml
import ast
import re
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from analyse_components import utils

# Define regex patterns here so they are isolated and compiled once.
_META_PATTERNS: Dict[str, re.Pattern] = {
    "dataset": re.compile(r"dataset:\s+(.+)", re.IGNORECASE),
    "max_eval": re.compile(r"eval:\s+(\d+)", re.IGNORECASE),
    "epochs": re.compile(r"epochs:\s+(\d+)", re.IGNORECASE),
    "modules": re.compile(r"mod_list:\s+(\[.+\])", re.IGNORECASE),
    "seed": re.compile(r"seed:\s+(\d+)", re.IGNORECASE),
    "dataset_out": re.compile(r"DATASET NAME:\s+(.+)", re.IGNORECASE),
    "epochs_out": re.compile(r"EPOCHS FOR TRAINING:\s+(\d+)", re.IGNORECASE),
    "total_time": re.compile(r"TOTAL TIME -------->\s*([0-9]*\.?[0-9]+)", re.IGNORECASE),
    # Changed to capture the group directly in findall
    "score": re.compile(r"Score:\s+\.?(-*[0-9]*\.?[0-9]+)")
}

def _get_config_metadata(experiment_dir: Path) -> Tuple[Optional[int], Optional[str]]:
    """
    Get epochs and dataset info ONLY from config.yaml or fallback parsing.
    Does NOT retrieve scores/losses (separated concerns).
    """
    config_file = experiment_dir / 'config.yaml'
    epochs, dataset = None, None

    # 1. Preferred method: config.yaml
    if config_file.exists():
        try:
            with open(config_file, 'r') as f:
                config_data = yaml.safe_load(f)
            epochs = config_data.get('epochs')
            dataset = config_data.get('dataset')
            # If found, return immediately
            if epochs is not None or dataset is not None:
                return epochs, dataset
        except Exception as e:
            logging.warning("Could not read %s: %s", config_file, e)

    # 2. Fallback method: .out or .log file
    try:
        out_files = list(experiment_dir.glob('*.out'))
        if not out_files:
            out_files = list(experiment_dir.glob('*.log'))

        if not out_files:
            return None, None
        
        out_file_path = out_files[0]

        with open(out_file_path, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
            m_dataset = _META_PATTERNS["dataset_out"].search(content)
            m_epochs = _META_PATTERNS["epochs_out"].search(content)
            
            dataset = m_dataset.group(1).strip() if m_dataset else None
            epochs = int(m_epochs.group(1)) if m_epochs else None
            
            return epochs, dataset
    except Exception as e:
        logging.error("Failed to read .out/.log file in %s: %s", experiment_dir.name, e)
        return None, None
    
def parse_final_losses_from_log(log_path: Path) -> List[float]:
    """
    Extracts the final 'loss' for each network by analyzing the .out/.log file.
    It looks for the 'ACCURACY:' line and backtracks to find the validation loss.
    """
    losses = []
    
    try:
        with open(log_path, 'r', encoding='utf-8', errors='ignore') as f:
            lines = f.readlines()
            
        # Regex to capture numeric value after "loss:"
        loss_pattern = re.compile(r"loss:\s+([0-9.]+)\s+-\s+accuracy:")

        for i, line in enumerate(lines):
            # 1. Find the anchor marking the end of a network training
            if "ACCURACY:" in line:
                
                # 2. Look backwards to find the Keras output line
                found_loss = False
                # Check previous 15 lines
                for j in range(i - 1, max(0, i - 15), -1):
                    prev_line = lines[j].strip()
                    
                    if not prev_line or "Restoring model weights" in prev_line or "early stopping" in prev_line:
                        continue
                    
                    # 3. Match the loss pattern
                    match = loss_pattern.search(prev_line)
                    if match:
                        loss_val = float(match.group(1))
                        losses.append(loss_val)
                        found_loss = True
                        break
                
                if not found_loss:
                    losses.append(None)
                    
    except Exception as e:
        print(f"Error parsing losses from {log_path}: {e}")
        return []

    return losses

def parse_scores_from_log(log_path: Path) -> List[float]:
    """
    Extracts all 'Score: ...' occurrences from the log file.
    """
    scores = []
    try:
        with open(log_path, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
        # findall returns a list of strings matching the group
        scores_str = _META_PATTERNS["score"].findall(content)
        scores = [float(s) for s in scores_str]
    except Exception as e:
        logging.error("Error parsing scores from %s: %s", log_path.name, e)
    return scores

def parse_experiment_data(experiment_dir: Path, root_path: Path) -> List[Dict[str, Any]]:
    """
    Parses experiment logs. Returns a list of network data dictionaries.
    """
    logs_dir = experiment_dir / 'algorithm_logs'
    acc_file = logs_dir / 'acc_report.txt'
    quantize = logs_dir / 'quantization_report.txt'
    flop_params_file = experiment_dir / 'flops_report.txt'

    # --- 1. METADATA (Epochs/Dataset) ---
    # Try parsing from folder name first
    parsed_name_data = utils.parse_experiment_name(experiment_dir.name)
    epochs = parsed_name_data.get('Epochs')
    dataset = parsed_name_data.get('Dataset')

    # Fallback to config/log if folder name parsing failed
    if epochs is None or dataset is None:
        epochs_fb, dataset_fb = _get_config_metadata(experiment_dir)
        if epochs is None: epochs = epochs_fb
        if dataset is None: dataset = dataset_fb

    # --- 2. LOG DATA (Losses/Scores) ---
    # We need to read the log file regardless of how we got the metadata
    losses = []
    scores = []
    
    # Find the log file
    out_files = list(experiment_dir.glob('*.out'))
    if not out_files: out_files = list(experiment_dir.glob('*.log'))
    
    if out_files:
        log_file = out_files[0]
        # Extract Losses
        losses = parse_final_losses_from_log(log_file)
    else:
        logging.warning("No .out/.log file found in %s. Losses/Scores will be None.", experiment_dir.name)

    # --- 3. CORE DATA (Accuracy/Hyperparams) ---
    try:
        with open(acc_file, 'r') as f_acc:
            acc_lines = f_acc.readlines()
        flops_params = []
        if flop_params_file.exists():
            with open(flop_params_file) as f_flops:
                flops_params = f_flops.readlines()    
        quantize_lines = []
        if quantize.exists():
            with open(quantize, 'r') as f_quantize:
                quantize_lines = f_quantize.readlines()
    except Exception as e:
        logging.error("Could not read log files in %s: %s", logs_dir, e)
        return []

    experiment_networks = []
    
    # Iterate using index to safely access all lists (handling different lengths)
    min_len = len(acc_lines)
    
    for i in range(min_len):
        try:
            acc_line = acc_lines[i]
            if acc_line.strip() == 'None':
                accuracy = 0.0
            else:
                accuracy = float(acc_line.strip())
            
            # Handle Quantization
            acc_quant = accuracy # Default to normal accuracy
            if quantize.exists() and i < len(quantize_lines):
                try:
                    # Expected format: "original_acc, quantized_acc"
                    parts = quantize_lines[i].strip().split(',')
                    if len(parts) > 1:
                        acc_quant = float(parts[1])
                except ValueError:
                    pass 
            flops = 0
            if flop_params_file.exists() and i < len(flops_params):
                params_line = flops_params[i]
                parts = params_line.strip().split(' ')
                if len(parts) > 1:
                    flops = float(parts[1])

            # Handle Losses (safe access)
            loss_val = losses[i] if i < len(losses) else None
            
            # Handle Scores (safe access)
            score_val = -0.77*accuracy-0.33*(1-flops/150000000) if accuracy > 0.0 else 100000 # Default scoring formula

            network_data = {
                'epochs': epochs,
                'dataset': dataset,
                'accuracy': accuracy,
                # 'hyperparams': hyper_dict,
                'experiment_source': str(experiment_dir.relative_to(root_path)), 
                'accuracy_quantization': acc_quant,
                'flops': flops,
                'score': score_val,
                'loss': loss_val
            }
            experiment_networks.append(network_data)
            
        except Exception as e:
            logging.error("Failed to parse row %d in %s: %s", i+1, experiment_dir.name, e)

    return experiment_networks