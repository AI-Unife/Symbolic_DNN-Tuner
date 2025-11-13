#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Pannello di controllo per l'analisi degli esperimenti.

Orchestra i moduli di parsing, database e analisi per generare
i report CSV richiesti.
"""

import argparse
import logging
from pathlib import Path

# Importa i TUOI moduli personalizzati
from analyse_components import parsing
from analyse_components import database
from analyse_components import analysis
from analyse_components import utils

def setup_logging(verbosity: int) -> None:
    # (Codice per configurare il logging... identico a prima)
    level = logging.WARNING
    if verbosity == 1: level = logging.INFO
    elif verbosity >= 2: level = logging.DEBUG
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%H:%M:%S",
    )

def parse_args() -> argparse.Namespace:
    """Parses command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Pannello di controllo per l'analisi dei risultati."
    )
    # Argomenti di base
    parser.add_argument("--base-dir", type=Path, default=Path("./"),
                        help="Cartella che contiene le sottocartelle degli esperimenti.")
    parser.add_argument("--exp-prefix", type=str, default="",
                        help="Elabora solo le cartelle che iniziano con questo prefisso.")
    
    # File di output
    parser.add_argument("--tested-model-csv", type=Path, default=Path("tested_model.csv"),
                        help="File di output per *ogni* rete testata.")
    parser.add_argument("--total-csv", type=Path, default=Path("total.csv"),
                        help="File di output per il riepilogo (una riga per esperimento).")
    parser.add_argument("--mean-csv", type=Path, default=Path("mean.csv"),
                        help="File di output per le medie.")
    
    # Flag per scegliere le analisi!
    parser.add_argument("--skip-plots", action="store_true",
                        help="Salta la generazione di grafici e analisi per-esperimento.")
    parser.add_argument("--skip-train", action="store_true",
                        help="Salta il retraining del modello (implica --no-plots se non specificato).")
    
    # Flag di verbosità
    parser.add_argument("-v", "--verbose", action="count", default=0,
                        help="Aumenta verbosità (-v per INFO, -vv per DEBUG).")
    return parser.parse_args()

def main():
    args = parse_args()
    setup_logging(args.verbose)
    
    logging.info("Avvio analisi, Base dir: %s", args.base_dir)

    # 1. Carica il DB esistente di 'tested_model.csv'
    db, all_hyper_keys = database.load_existing_db(args.tested_model_csv)

    # Liste per contenere i risultati
    total_experiment_summaries = []
    total_networks_scanned = 0

    # 2. Scansiona le cartelle
    for logs_dir in args.base_dir.rglob('algorithm_logs'):
        experiment_dir = logs_dir.parent
        if not utils.should_select_experiment_dir(experiment_dir.name, args.exp_prefix):
            continue
            
        logging.info("--- Analisi esperimento: %s ---", experiment_dir.relative_to(args.base_dir))

        # 3. Esegui analisi "pesanti" (opzionali)
        if not args.skip_plots:
            analysis.run_per_experiment_analysis(experiment_dir, 
                                                 run_train=(not args.skip_train))
        
        # 4. Parsing (sempre eseguito)
        network_data_list = parsing.parse_experiment_data(experiment_dir, args.base_dir)
        if not network_data_list:
            logging.warning("Nessun dato di rete trovato in %s, saltato.", experiment_dir.name)
            continue
        
        total_networks_scanned += len(network_data_list)

        # 5. Aggiorna il DB e crea il riepilogo 'total'
        summary_row, updated_count, new_count = database.update_model_db(db, network_data_list, all_hyper_keys)
        if summary_row:
            total_experiment_summaries.append(summary_row)
            
        if updated_count or new_count:
             logging.info("Unione 'tested_model.csv': %d record nuovi, %d aggiornati da %s",
                          new_count, updated_count, experiment_dir.name)

    # 6. Scrittura finale dei file
    
    # --- File 1: tested_model.csv ---
    db_records = list(db.values())
    database.write_tested_model_file(args.tested_model_csv, db_records, all_hyper_keys)
    print(f"✅ Creato file modelli testati: {args.tested_model_csv}")

    # --- File 2: total.csv ---
    df_total = database.write_total_file(args.total_csv, total_experiment_summaries)
    print(f"✅ Creato file riepilogo totale: {args.total_csv}")

    # --- File 3: mean.csv ---
    database.write_mean_file(args.mean_csv, df_total)
    print(f"✅ Creato file riepilogo medie: {args.mean_csv}")
    
    # --- Riepilogo Console ---
    print("\n--- Riepilogo Esecuzione ---")
    print(f"Reti totali analizzate nei file di log (scan attuale): {total_networks_scanned}")
    print(f"Reti uniche (totali, dopo unione e filtro):     {len(db_records)}")

if __name__ == "__main__":
    main()