# In database.py
import pandas as pd
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Any, Set
import numpy as np

# Importa le utility e le costanti necessarie
from analyse_components import utils # Conterrà extract_tuner
from analyse_components import parsing # Conterrà STATIC_HEADERS

# Definiamo le costanti qui
STATIC_HEADERS = ['epochs', 'dataset', 'accuracy']

def create_key_from_record(record: Dict[str, Any]) -> str:
    # (Codice identico a prima)
    key_parts = []
    key_parts.append(f"dataset:{record.get('dataset')}")
    key_parts.append(f"epochs:{record.get('epochs')}")
    hyper_dict = record.get('hyperparams', {})
    if not hyper_dict:
        hyper_dict = {k: v for k, v in record.items() if k not in STATIC_HEADERS}
    for k, v in sorted(hyper_dict.items()):
        key_parts.append(f"{k}:{str(v)}") 
    return "|".join(key_parts)

def load_existing_db(csv_path: Path) -> Tuple[Dict[str, Dict], Set[str]]:
    # (Codice identico a prima)
    db = {}
    all_hyper_keys: Set[str] = set()
    xlsx_path = csv_path.with_suffix('.xlsx')
    if xlsx_path.exists():
        df = pd.read_excel(xlsx_path)
        logging.info("Caricamento del database esistente da: %s", xlsx_path)
    elif csv_path.exists():
        df = pd.read_csv(csv_path)
        logging.info("Caricamento del database esistente da: %s", csv_path)
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
                logging.warning("Valore non valido ('%s' o '%s') nel file, riga saltata.",
                                row.get('accuracy'), row.get('epochs'))
                continue
            key = create_key_from_record(record)
            if key not in db or record['accuracy'] > db[key]['accuracy']:
                db[key] = record
    except Exception as e:
        logging.error("Impossibile caricare o analizzare %s: %s", csv_path.name, e)
        logging.info("L'analisi continuerà senza i dati precedenti.")
        return {}, set()
    logging.info("Caricati %d record unici esistenti.", len(db))
    return db, all_hyper_keys

def update_model_db(db: Dict[str, Dict], network_data_list: List[Dict], all_hyper_keys: Set[str]) -> Tuple[Dict, int, int]:
    """
    Aggiorna il DB in memoria con una lista di nuove reti.
    Restituisce (summary_row, update_count, new_count)
    """
    if not network_data_list:
        return {}, 0, 0
        
    update_count = 0
    new_count = 0
    
    # Trova la rete migliore in questa lista specifica
    best_network = max(network_data_list, key=lambda x: x['accuracy'])
    
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
            
    # Crea il riepilogo per total.csv
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
    """Helper interna per la scrittura di file."""
    xlsx_path = csv_path.with_suffix('.xlsx')
    logging.info("Scrittura di %d record aggregati in: %s (e .xlsx)", len(data), csv_path)
    
    if not data:
        logging.warning("Nessun dato da scrivere per %s.", csv_path.name)
        df_out = pd.DataFrame(columns=headers)
    else:
        df_out = pd.DataFrame(data, columns=headers)

    try:
        df_out.to_csv(csv_path, index=False)
        df_out.to_excel(xlsx_path, index=False)
    except IOError as e:
        logging.error("Impossibile scrivere i file di output (%s): %s", csv_path.name, e)

def write_tested_model_file(csv_path: Path, db_records: List[Dict], all_hyper_keys: Set[str]):
    """Appiattisce e scrive il database 'tested_model.csv'."""
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
    """Scrive 'total.csv' e restituisce il DataFrame per 'mean.csv'."""
    if not total_summaries:
        _write_file(csv_path, [], [])
        return pd.DataFrame()
        
    df_total = pd.DataFrame(total_summaries)
    static_cols = ['Experiment Name', 'Tuner', 'Dataset', 'Epochs', 'Eval Count', 'Best Accuracy']
    dynamic_cols = sorted([c for c in df_total.columns if c not in static_cols])
    total_headers = static_cols + dynamic_cols
    
    _write_file(csv_path, df_total.to_dict('records'), total_headers)
    return df_total

def write_mean_file(csv_path: Path, df_total: pd.DataFrame):
    """Calcola e scrive 'mean.csv' da df_total."""
    if df_total.empty:
        logging.warning("DataFrame 'total' vuoto, impossibile generare 'mean.csv'.")
        _write_file(csv_path, [], [])
        return
        
    if "Tuner" not in df_total.columns or "Dataset" not in df_total.columns:
        logging.warning("Colonne 'Tuner' o 'Dataset' non trovate. Impossibile generare 'mean.csv'.")
        return
        
    numeric_cols = df_total.select_dtypes(include=[np.number]).columns.tolist()
    metrics = [c for c in numeric_cols if c in ('Best Accuracy', 'Eval Count', 'Epochs')]
    if not metrics:
        logging.warning("Nessuna colonna metrica trovata per il riepilogo 'mean'.")
        return

    work = df_total[["Dataset", "Tuner"] + metrics].copy()
    group_keys = ["Dataset", "Tuner"]
    grouped = work.groupby(group_keys, dropna=False)
    logging.info("🔍 Generazione medie: Raggruppando per %s, %d gruppi trovati.", group_keys, len(grouped))
    
    agg = grouped[metrics].agg(["mean", "std"])
    agg.columns = [f"{m}_{stat}" for (m, stat) in agg.columns]
    agg = agg.sort_index()
    
    # Scrive il file mean
    agg.to_csv(csv_path, index=True)
    agg.to_excel(csv_path.with_suffix(".xlsx"), index=True)