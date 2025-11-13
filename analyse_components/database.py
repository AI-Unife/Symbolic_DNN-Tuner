# In database.py
import pandas as pd
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Set
import numpy as np

# Import utilities and constants
from analyse_components import utils 
from analyse_components import parsing

# Define constants here
STATIC_HEADERS = ['epochs', 'dataset', 'accuracy']

def create_key_from_record(record: Dict[str, Any]) -> str:
    """
    Creates a unique, sorted string key from a data record based on all
    parameters *except* accuracy. Used for de-duplication.
    """
    key_parts = []
    key_parts.append(f"dataset:{record.get('dataset')}")
    key_parts.append(f"epochs:{record.get('epochs')}")
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
    
    Returns a tuple:
    1.  db: A dict mapping {unique_key: best_record_so_far}
    2.  all_hyper_keys: A set of all hyperparameter column names found.
    """
    db = {}
    all_hyper_keys: Set[str] = set()
    xlsx_path = csv_path.with_suffix('.xlsx')
    if xlsx_path.exists():
        df = pd.read_excel(xlsx_path)
        logging.info("Loading existing database from: %s", xlsx_path)
    elif csv_path.exists():
        df = pd.read_csv(csv_path)
        logging.info("Loading existing database from: %s", csv_path)
    else:
        return db, all_hyper_keys
        
    df = df.where(pd.notna(df), None)
    
    try:
        file_hyper_keys = [h for h in df.columns if h not in STATIC_HEADERS]
        all_hyper_keys.update(file_hyper_keys)
        
        for row in df.to_dict('records'):
            record = {}
            hyper_params = {}
            for header in STATIC_HEADERS:
                record[header] = row.get(header)
            for h_key in file_hyper_keys:
                hyper_params[h_key] = row.get(h_key)
            record['hyperparams'] = hyper_params
            
            try:
                record['accuracy'] = float(row['accuracy'])
                if 'epochs' in record and record['epochs'] is not None:
                     record['epochs'] = int(float(record['epochs']))
            except (ValueError, TypeError):
                logging.warning("Invalid value ('%s' or '%s') in file, skipping row.",
                                row.get('accuracy'), row.get('epochs'))
                continue
            
            key = create_key_from_record(record)
            if key not in db or record['accuracy'] > db[key]['accuracy']:
                db[key] = record
                    
    except Exception as e:
        logging.error("Failed to load or parse %s: %s", csv_path.name, e)
        logging.info("Analysis will continue without previous data.")
        return {}, set()
        
    logging.info("Loaded %d existing unique records.", len(db))
    return db, all_hyper_keys

def update_model_db(db: Dict[str, Dict], network_data_list: List[Dict], all_hyper_keys: Set[str]) -> Tuple[Dict, int, int]:
    """
    Updates the in-memory DB with a list of new networks from one experiment.
    
    Returns:
    - (summary_row): A dict summarizing this experiment for 'total.csv'.
    - (update_count): Number of models that updated their accuracy.
    - (new_count): Number of new model configurations found.
    """
    if not network_data_list:
        return {}, 0, 0
        
    update_count = 0
    new_count = 0
    
    # Find the best network *in this specific list*
    best_network = max(network_data_list, key=lambda x: x['accuracy'])
    
    # Update the global database (db)
    for network_data in network_data_list:
        all_hyper_keys.update(network_data['hyperparams'].keys())
        key = create_key_from_record(network_data)
        new_accuracy = network_data['accuracy']
        
        if key in db:
            stored_accuracy = db[key]['accuracy']
            if new_accuracy > stored_accuracy:
                db[key] = network_data
                update_count += 1
        else:
            db[key] = network_data
            new_count += 1
            
    # Create the summary row for total.csv
    summary_row = {
        'Experiment Name': Path(best_network['experiment_source']).name,
        'Tuner': utils.extract_tuner(Path(best_network['experiment_source']).name),
        'Dataset': best_network['dataset'],
        'Epochs': best_network['epochs'],
        'Eval Count': len(network_data_list),
        'Best Accuracy': best_network['accuracy'],
    }
    summary_row.update(best_network['hyperparams'])
    
    return summary_row, update_count, new_count

def _write_file(csv_path: Path, data: List[Dict], headers: List[str]):
    """Internal helper function for writing files."""
    xlsx_path = csv_path.with_suffix('.xlsx')
    logging.info("Writing %d records to: %s (and .xlsx)", len(data), csv_path)
    
    if not data:
        logging.warning("No data to write for %s.", csv_path.name)
        df_out = pd.DataFrame(columns=headers)
    else:
        df_out = pd.DataFrame(data, columns=headers)

    try:
        df_out.to_csv(csv_path, index=False)
        df_out.to_excel(xlsx_path, index=False)
    except IOError as e:
        logging.error("Failed to write output files (%s): %s", csv_path.name, e)

def write_tested_model_file(csv_path: Path, db_records: List[Dict], all_hyper_keys: Set[str]):
    """Flattens and writes the 'tested_model.csv' database."""
    flat_db_records = []
    sorted_hyper_keys = sorted(list(all_hyper_keys))
    tested_model_headers = STATIC_HEADERS + sorted_hyper_keys
    
    for record in db_records:
        row = {}
        for header in STATIC_HEADERS:
            row[header] = record.get(header)
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
        
    df_total = pd.DataFrame(total_summaries)
    # Ensure a consistent column order
    static_cols = ['Experiment Name', 'Tuner', 'Dataset', 'Epochs', 'Eval Count', 'Best Accuracy']
    dynamic_cols = sorted([c for c in df_total.columns if c not in static_cols])
    total_headers = static_cols + dynamic_cols
    
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
        
    numeric_cols = df_total.select_dtypes(include=[np.number]).columns.tolist()
    metrics = [c for c in numeric_cols if c in ('Best Accuracy', 'Eval Count', 'Epochs')]
    if not metrics:
        logging.warning("No numeric metric columns found for 'mean' summary.")
        return

    work = df_total[["Dataset", "Tuner"] + metrics].copy()
    group_keys = ["Dataset", "Tuner"]
    grouped = work.groupby(group_keys, dropna=False)
    logging.info("🔍 Generating means: Grouping by %s, %d groups found.", group_keys, len(grouped))
    
    # Calculate mean/std
    agg = grouped[metrics].agg(["mean", "std"])
    agg.columns = [f"{m}_{stat}" for (m, stat) in agg.columns] # Flatten multi-index
    agg = agg.sort_index()
    
    # Write the mean file
    agg.to_csv(csv_path, index=True)
    agg.to_excel(csv_path.with_suffix(".xlsx"), index=True)