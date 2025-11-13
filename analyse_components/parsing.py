# In parsing.py
import yaml
import ast
import re
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

# Define regex patterns here so they are isolated
_META_PATTERNS: Dict[str, re.Pattern] = {
    "dataset": re.compile(r"dataset:\s+(.+)", re.IGNORECASE),
    "max_eval": re.compile(r"eval:\s+(\d+)", re.IGNORECASE),
    "epochs": re.compile(r"epochs:\s+(\d+)", re.IGNORECASE),
    "modules": re.compile(r"mod_list:\s+(\[.+\])", re.IGNORECASE),
    "seed": re.compile(r"seed:\s+(\d+)", re.IGNORECASE),
    "dataset_out": re.compile(r"DATASET NAME:\s+(.+)", re.IGNORECASE),
    "epochs_out": re.compile(r"EPOCHS FOR TRAINING:\s+(\d+)", re.IGNORECASE),
    "total_time": re.compile(r"TOTAL TIME -------->\s*([0-9]*\.?[0-9]+)", re.IGNORECASE),
}

def _get_config_data(experiment_dir: Path) -> Tuple[Optional[int], Optional[str]]:
    """
    Get epochs and dataset info, trying config.yaml first, then falling
    back to parsing a .out file.
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
            if epochs is not None or dataset is not None:
                return epochs, dataset
        except Exception as e:
            logging.warning("Could not read %s: %s", config_file, e)

    # 2. Fallback method: .out file
    try:
        out_files = list(experiment_dir.glob('*.out'))
        if not out_files:
            if not config_file.exists():
                logging.warning("No config.yaml or .out file found in %s", experiment_dir.name)
            return None, None
        
        out_file_path = out_files[0]
        if len(out_files) > 1:
            logging.warning("Multiple .out files found; using %s for metadata", out_file_path.name)

        with open(out_file_path, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
            m_dataset = _META_PATTERNS["dataset_out"].search(content)
            m_epochs = _META_PATTERNS["epochs_out"].search(content)
            
            dataset = m_dataset.group(1).strip() if m_dataset else None
            epochs = int(m_epochs.group(1)) if m_epochs else None
            
            return epochs, dataset
    except Exception as e:
        logging.error("Failed to read .out file %s: %s", out_file_path.name, e)
        return None, None


def parse_experiment_data(experiment_dir: Path, root_path: Path) -> List[Dict[str, Any]]:
    """
    Parses the core log files (acc_report.txt, hyper-neural.txt)
    for a single experiment and returns a list of all networks tested.
    """
    logs_dir = experiment_dir / 'algorithm_logs'
    acc_file = logs_dir / 'acc_report.txt'
    hyper_file = logs_dir / 'hyper-neural.txt'

    if not (acc_file.exists() and hyper_file.exists()):
        logging.debug("Core log files missing in %s, skipping.", experiment_dir.name)
        return []

    epochs, dataset = _get_config_data(experiment_dir)

    try:
        with open(acc_file, 'r') as f_acc:
            acc_lines = f_acc.readlines()
        with open(hyper_file, 'r') as f_hyper:
            hyper_lines = f_hyper.readlines()
    except Exception as e:
        logging.error("Could not read log files in %s: %s", logs_dir, e)
        return []

    if len(acc_lines) != len(hyper_lines):
        logging.warning("%s: Log file line mismatch (%d vs %d). Processing minimum.",
                        experiment_dir.name, len(acc_lines), len(hyper_lines))
    
    experiment_networks = []
    
    for i, (acc_line, hyper_line) in enumerate(zip(acc_lines, hyper_lines)):
        try:
            accuracy = float(acc_line.strip())
            hyper_str = hyper_line.strip()
            if not hyper_str:
                logging.warning("Row %d in %s is empty. Skipped.", i+1, hyper_file)
                continue
            
            # Safely evaluate the hyperparameter dictionary string
            hyper_dict = ast.literal_eval(hyper_str)
            if not isinstance(hyper_dict, dict): 
                raise ValueError("Parsed hyperparameter data is not a dict")

            network_data = {
                'epochs': epochs,
                'dataset': dataset,
                'accuracy': accuracy,
                'hyperparams': hyper_dict,
                'experiment_source': str(experiment_dir.relative_to(root_path)) 
            }
            experiment_networks.append(network_data)
        except Exception as e:
            logging.error("Failed to parse row %d in %s: %s", i+1, experiment_dir.name, e)
            logging.error("   Problematic hyper-line: %s", hyper_line.strip())

    return experiment_networks