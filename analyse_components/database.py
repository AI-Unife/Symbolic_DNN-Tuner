# In database.py
"""
This module manages the data persistence and aggregation layer.

It handles:
- Loading and saving the 'tested_model.csv' (a persistent database
  of all unique networks ever evaluated).
- De-duplicating networks using a unique key.
- Updating the in-memory database with newly parsed results.
- Writing the final summary files:
    - tested_model.csv: One row per unique network configuration.
    - total.csv: One row per experiment, showing the *best* network.
    - mean.csv: Aggregated means/stds grouped by Tuner and Dataset.
- (NEW) Writing the 'results.csv' summary for a single experiment.
"""

import pandas as pd
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Set
import numpy as np

# Import utilities and constants
from analyse_components import utils 
# Note: 'parsing' is imported but not used, could be removed.
# from analyse_components import parsing 

# Define constants here
# These are the static, non-hyperparameter columns.
STATIC_HEADERS = ['epochs', 'dataset', 'accuracy']

def create_key_from_record(record: Dict[str, Any]) -> str:
    """
    Creates a unique, sorted string key from a data record based on all
    parameters *except* accuracy. Used for de-duplication.

    This ensures that two networks with the same hyperparameters but
    different accuracies (e.g., from different runs) are treated
    as the same network configuration.

    Args:
        record: A network data dictionary.

    Returns:
        A unique string key (e.g., "dataset:CIFAR|epochs:50|lr:0.01|...")
    """
    key_parts = []
    key_parts.append(f"dataset:{record.get('dataset')}")
    key_parts.append(f"epochs:{record.get('epochs')}")
    
    # Get hyperparams, either from a nested dict or flat keys
    hyper_dict = record.get('hyperparams', {})
    if not hyper_dict:
        # Fallback for records already flattened (e.g., from CSV)
        hyper_dict = {k: v for k, v in record.items() if k not in STATIC_HEADERS}
        
    for k, v in sorted(hyper_dict.items()):
        key_parts.append(f"{k}:{str(v)}") 
        
    return "|".join(key_parts)

def load_existing_db(csv_path: Path) -> Tuple[Dict[str, Dict], Set[str]]:
    """
    Loads an existing tested_model.csv file (XLSX or CSV) into an in-memory database.
    
    The DB is a dictionary mapping {unique_key: best_record}.
    This efficiently handles de-duplication and ensures we only
    keep the record with the *highest* accuracy for any given
    hyperparameter set.

    Args:
        csv_path: The Path object for 'tested_model.csv'.

    Returns:
        A tuple:
        1.  db: A dict mapping {unique_key: best_record_so_far}
        2.  all_hyper_keys: A set of all hyperparameter column names found.
    """
    db = {}
    all_hyper_keys: Set[str] = set()
    
    # Prefer loading the .xlsx file if it exists, as it may be more robust
    xlsx_path = csv_path.with_suffix('.xlsx')
    
    df: Optional[pd.DataFrame] = None
    load_path: Optional[Path] = None

    if xlsx_path.exists():
        df = pd.read_excel(xlsx_path)
        load_path = xlsx_path
    elif csv_path.exists():
        df = pd.read_csv(csv_path)
        load_path = csv_path
    else:
        # No existing DB found, return empty state
        return db, all_hyper_keys
    
    logging.info("Loading existing database from: %s", load_path)
        
    # Replace Pandas' NaT/NaN with None for consistency
    df = df.where(pd.notna(df), None)
    
    try:
        # Identify which columns are hyperparameters
        file_hyper_keys = [h for h in df.columns if h not in STATIC_HEADERS]
        all_hyper_keys.update(file_hyper_keys)
        
        # Iterate over each row in the loaded file
        for row in df.to_dict('records'):
            record = {}
            hyper_params = {}
            
            # Re-build the nested structure
            for header in STATIC_HEADERS:
                record[header] = row.get(header)
            for h_key in file_hyper_keys:
                hyper_params[h_key] = row.get(h_key)
            record['hyperparams'] = hyper_params
            
            # Basic type-casting and validation
            try:
                record['accuracy'] = float(row['accuracy'])
                if 'epochs' in record and record['epochs'] is not None:
                     record['epochs'] = int(float(record['epochs']))
            except (ValueError, TypeError):
                logging.warning("Invalid value ('%s' or '%s') in file, skipping row.",
                                row.get('accuracy'), row.get('epochs'))
                continue
            
            # Create the unique key for this network configuration
            key = create_key_from_record(record)
            
            # Update the DB: only keep this record if it's new
            # or has a better accuracy than the one already stored.
            if key not in db or record['accuracy'] > db[key]['accuracy']:
                db[key] = record
                    
    except Exception as e:
        logging.error("Failed to load or parse %s: %s", (load_path or csv_path.name), e)
        logging.info("Analysis will continue without previous data.")
        return {}, set()
        
    logging.info("Loaded %d existing unique records.", len(db))
    return db, all_hyper_keys

def update_model_db(network_data_list: List[Dict]) -> Tuple[Dict, int, int]:
    """
    Updates the in-memory DB with a list of new networks from one experiment.
    
    This function also identifies the single best network *from this
    experiment* and returns it as a 'summary_row'.

    Args:
        db: The global in-memory database {key: record}.
        network_data_list: A list of new network records from parse_experiment_data.
        all_hyper_keys: The global set of all param names (will be updated).

    Returns:
        A tuple:
        - (summary_row): A dict summarizing this experiment for 'total.csv'.
        - (update_count): Number of models that updated their accuracy.
        - (new_count): Number of new model configurations found.
    """
    if not network_data_list:
        return {}, 0, 0

    # Find the best network *in this specific list* (from one experiment)
    # x[0] è l'indice, x[1] è il dizionario vero e proprio
    idx, best_network = min(enumerate(network_data_list), key=lambda x: x[1]['score'])

        
   
    # Create the summary row for total.csv
    # This row is based *only* on the best network from this experiment,
    # but metadata (Tuner, Dataset, Epochs) comes from the FOLDER NAME.
    exp_name = Path(best_network['experiment_source']).name
    parsed_name_data = utils.parse_experiment_name(exp_name)
    
    summary_row = {
        'Base Dir': str(Path(best_network['experiment_source']).parent),
        'Experiment Name': exp_name,
        'Tuner': parsed_name_data.get('Tuner'),     # From folder name
        'Dataset': parsed_name_data.get('Dataset'), # From folder name
        'Eval Count': idx,     # How many models this experiment tested
        'Best Score': best_network['score'],
        'Best Accuracy': best_network['accuracy'],
        'Best FLOPs': best_network.get('flops', 0),
        'Best Quantized': best_network.get('accuracy_quantization', None),
        'Best Latency': best_network.get('latency', 0)
    }

    
    return summary_row

def _get_ordered_headers(data: List[Dict], static_cols: List[str]) -> List[str]:
    """Helper to get a consistent column order for CSVs."""
    if not data:
        return static_cols
        
    # Get all dynamic (hyperparameter) columns from the data
    all_keys = set()
    for row in data:
        all_keys.update(row.keys())
        
    dynamic_cols = sorted([c for c in all_keys if c not in static_cols])
    return static_cols + dynamic_cols

def _write_file(csv_path: Path, data: List[Dict], headers: List[str]):
    """
    Internal helper function for writing files to both .csv and .xlsx.
    """
    xlsx_path = csv_path.with_suffix('.xlsx')
    logging.info("Writing %d records to: %s (and .xlsx)", len(data), csv_path)
    
    if not data:
        logging.warning("No data to write for %s.", csv_path.name)
        df_out = pd.DataFrame(columns=headers)
    else:
        # Use columns=headers to ensure consistent order and all columns
        df_out = pd.DataFrame(data, columns=headers)

    try:
        df_out.to_csv(csv_path, index=False)
        df_out.to_excel(xlsx_path, index=False)
    except IOError as e:
        logging.error("Failed to write output files (%s): %s", csv_path.name, e)

def write_tested_model_file(csv_path: Path, db_records: List[Dict], all_hyper_keys: Set[str]):
    """Flattens and writes the 'tested_model.csv' database."""
    flat_db_records = []
    # Get a stable, sorted list of all hyperparameter names
    sorted_hyper_keys = sorted(list(all_hyper_keys))
    tested_model_headers = STATIC_HEADERS + sorted_hyper_keys
    
    for record in db_records:
        row = {}
        # Add static columns
        for header in STATIC_HEADERS:
            row[header] = record.get(header)
        
        # Add dynamic hyperparameter columns
        hyper_dict = record.get('hyperparams', {})
        for key in sorted_hyper_keys:
            row[key] = hyper_dict.get(key)
            
        flat_db_records.append(row)
        
    _write_file(csv_path, flat_db_records, tested_model_headers)

def write_total_file(csv_path: Path, total_summaries: List[Dict]) -> pd.DataFrame:
    """Writes 'total.csv' and returns the DataFrame for use in 'mean.csv'."""
    if not total_summaries:
        _write_file(csv_path, [], [])
        return pd.DataFrame()
        
    # Define a consistent column order for the 'total' summary
    static_cols = ['Base Dir', 'Experiment Name', 'Tuner', 'Dataset', 'Epochs', 'Eval Count', 'Best Accuracy']
    total_headers = _get_ordered_headers(total_summaries, static_cols)
    
    # Pass the data and headers to the writer
    df_total = pd.DataFrame(total_summaries)
    _write_file(csv_path, df_total.to_dict('records'), total_headers)
    
    return df_total

def write_mean_file(csv_path: Path, df_total: pd.DataFrame):
    """Calculates and writes 'mean.csv' from the 'total' DataFrame."""
    if df_total.empty:
        logging.warning("'total' DataFrame is empty, cannot generate 'mean.csv'.")
        _write_file(csv_path, [], [])
        return
        
    if "Tuner" not in df_total.columns or "Dataset" not in df_total.columns:
        logging.warning("'Tuner' or 'Dataset' columns not found. Cannot generate 'mean.csv'.")
        return
        
    # Identify all numeric columns to aggregate
    numeric_cols = df_total.select_dtypes(include=[np.number]).columns.tolist()
    # Define which metrics we want to average
    metrics = [c for c in numeric_cols if c in ( 'Best Score', 'Best Accuracy', 'Eval Count', 'Epochs', 'Best Quantized','Best FLOPs', 'Best Latency')]
    if not metrics:
        logging.warning("No numeric metric columns found for 'mean' summary.")
        return

    work = df_total[["Dataset", "Tuner"] + metrics].copy()
    group_keys = ["Dataset", "Tuner"]
    
    # Group by Dataset and Tuner
    grouped = work.groupby(group_keys, dropna=False)
    logging.info("🔍 Generating means: Grouping by %s, %d groups found.", group_keys, len(grouped))
    
    # Calculate mean and standard deviation
    agg = grouped[metrics].agg(["mean", "std"])
    
    # Flatten the multi-index columns (e.g., ('Best Accuracy', 'mean') -> 'Best Accuracy_mean')
    agg.columns = [f"{m}_{stat}" for (m, stat) in agg.columns] 
    agg = agg.sort_index()
    
    # Write the mean file (don't use _write_file, this one needs the index)
    try:
        agg.to_csv(csv_path, index=True)
        agg.to_excel(csv_path.with_suffix(".xlsx"), index=True)
    except IOError as e:
        logging.error("Failed to write mean file (%s): %s", csv_path.name, e)

# --- NEW FUNCTION ---
def write_individual_experiment_summary(data: Dict, csv_path: Path):
    """
    Writes the single summary row for one experiment to its
    own 'results.csv' file.
    
    Args:
        summary_row: The summary dictionary created in update_model_db.
        csv_path: The full path to the output file (e.g., .../exp_name/results.csv).
    """
    if not data:
        logging.warning("No summary row provided, cannot write %s", csv_path)
        return

    # Use the internal _write_file helper to get both CSV and XLSX
    try:
        xlsx_path = csv_path.with_suffix('.xlsx')
        df_out = pd.DataFrame(data)
        df_out.to_csv(csv_path, index=False)
        df_out.to_excel(xlsx_path, index=False)
    except IOError as e:
        logging.error("Failed to write output files (%s): %s", csv_path.name, e)