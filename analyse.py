#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Main control panel for experiment analysis.

This script orchestrates the parsing, database, and analysis modules
to generate the requested CSV/XLSX reports.

It operates in two modes:

1.  DEFAULT (Full Analysis):
    -   Scans all experiment directories.
    -   Parses `algorithm_logs` to get all tested networks.
    -   (Optional) Runs per-experiment plots and training.
    -   Merges results with the existing `tested_model.csv`.
    -   SAVES/OVERWRITES `<experiment_dir>/results.csv` with a summary
      of the *best* model for that experiment.
    -   SAVES/OVERWRITES `total.csv` and `mean.csv` with a
      full summary from the *current run*.

2.  AGGREGATE-ONLY (`--aggregate-only`):
    -   Skips all parsing and optional analysis.
    -   Scans for existing `<experiment_dir>/results.csv` files.
    -   Builds `total.csv` and `mean.csv` by aggregating
      these individual summary files.
    -   This is much faster if you only want to update the
      master summary files.
"""

import argparse
import logging
import pandas as pd
from pathlib import Path
from typing import List, Dict, Any

# Import your custom modules
from analyse_components import parsing
from analyse_components import database
from analyse_components import analysis
from analyse_components import utils

def setup_logging(verbosity: int) -> None:
    """Configure logging level and format based on -v flags."""
    level = logging.WARNING
    if verbosity == 1:
        level = logging.INFO
    elif verbosity >= 2:
        level = logging.DEBUG
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%H:%M:%S",
    )

def parse_args() -> argparse.Namespace:
    """Parses command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Control panel for experiment results analysis."
    )
    # Basic arguments
    parser.add_argument("--base-dir", type=Path, default=Path("./"),
                        help="Folder containing the experiment subfolders.")
    parser.add_argument("--exp-prefix", type=str, default="",
                        help="Only process folders that start with this prefix.")
    
    # Output files
    parser.add_argument("--tested-model-csv", type=Path, default=Path("tested_model.csv"),
                        help="Output file for *every* tested network (one row per model).")
    parser.add_argument("--total-csv", type=Path, default=Path("total.csv"),
                        help="Output file for the summary (one row per experiment).")
    parser.add_argument("--mean-csv", type=Path, default=Path("mean.csv"),
                        help="Output file for the grouped means (one row per tuner/dataset).")
    
    # Flags to select analyses
    parser.add_argument("--plots", action="store_false",
                        help="Generating per-experiment plots and analyses.")
    parser.add_argument("--train", action="store_false",
                        help="Re-training the best model.")
    
    # NEW: Flag for fast aggregation
    parser.add_argument("--aggregate-only", action="store_true",
                        help="Skip parsing and aggregate existing 'results.csv' files.")
    
    # Verbosity flag
    parser.add_argument("-v", "--verbose", action="count", default=0,
                        help="Increase verbosity (-v for INFO, -vv for DEBUG).")
    return parser.parse_args()

def aggregate_results_from_csvs(args: argparse.Namespace) -> None:
    """
    Fast aggregation mode. Finds all 'results.csv' files and builds
    total.csv and mean.csv from them, skipping all parsing.
    """
    logging.info("--- AGGREGATE-ONLY Mode ---")
    total_experiment_summaries: List[Dict[str, Any]] = []
    
    # Use rglob to find all individual summary files
    summary_files_found = 0
    for results_csv_path in args.base_dir.rglob('algorithm_logs'):
        experiment_dir = results_csv_path.parent
        
        # Ensure we respect the experiment prefix
        if not utils.should_select_experiment_dir(experiment_dir.name, args.exp_prefix):
            continue
            
        logging.info("Aggregating: %s", results_csv_path)
        summary_files_found += 1
        try:
            # --- MODIFIED AGGREGATION LOGIC ---
            # 1. Read the MULTI-ROW results.csv
            df_exp = pd.read_csv(results_csv_path)
            if df_exp.empty:
                logging.warning("Empty results.csv in %s, skipping.", experiment_dir.name)
                continue
                
            # 2. Find the best network row *in this file*
            if 'accuracy' not in df_exp.columns:
                logging.error("No 'accuracy' column in %s, skipping.", results_csv_path)
                continue
            best_network_row = df_exp.loc[df_exp['score'].idxmin()]
            
            # 3. Parse metadata from the FOLDER NAME
            exp_name = experiment_dir.name
            parsed_name_data = utils.parse_experiment_name(exp_name)

            # 4. Identify hyperparameter columns (anything not static)
            static_cols = ['epochs', 'dataset', 'accuracy', 'experiment_source'] 
            hyper_cols = [c for c in df_exp.columns if c not in static_cols]

            # 5. Build the summary row for total.csv
            summary_row = {
                'Experiment Name': exp_name,
                'Tuner': parsed_name_data.get('Tuner'),
                'Dataset': parsed_name_data.get('Dataset'),
                'Epochs': parsed_name_data.get('Epochs'),
                'Eval Count': len(df_exp), # Total networks tested in this run
                'Best Score': best_network_row['score'],
                'Best Accuracy': best_network_row['accuracy'],
                'Best Quantized': best_network_row.get('accuracy_quantization', None),
                'Best FLOPs': best_network_row.get('flops', 0),
                'Best Latency': best_network_row.get('latency', 0)
                
            }
            
            # 6. Add hyperparams *of the best network*
            for col in hyper_cols:
                summary_row[col] = best_network_row.get(col)
                
            total_experiment_summaries.append(summary_row)
            # --- END MODIFICATION ---
        except Exception as e:
            logging.error("Failed to read or parse %s: %s", results_csv_path, e)

    logging.info("Found and aggregated %d 'results.csv' files.", summary_files_found)
    
    if not total_experiment_summaries:
        logging.warning("No 'results.csv' files found or parsed. No output generated.")
        return

    # --- Write output files ---
    
    # --- File 1: total.csv ---
    df_total = database.write_total_file(args.total_csv, total_experiment_summaries)
    print(f"✅ Created total summary file: {args.total_csv}")

    # --- File 2: mean.csv ---
    database.write_mean_file(args.mean_csv, df_total)
    print(f"✅ Created mean summary file: {args.mean_csv}")
    
    # --- Console Summary ---
    print("\n--- Aggregation Summary ---")
    print(f"Total experiments aggregated: {len(total_experiment_summaries)}")


def main():
    args = parse_args()
    setup_logging(args.verbose)
    
    # --- NEW: Check for --aggregate-only mode ---
    # If set, run the aggregation function and exit.
    if args.aggregate_only:
        aggregate_results_from_csvs(args)
        return
        
    # --- Standard Full Analysis Mode ---
    logging.info("Starting full analysis, Base dir: %s", args.base_dir)

    # 1. Load the existing 'tested_model.csv' database
    # This DB stores *every* network configuration ever tested,
    # mapping a unique key (params) to its best-known accuracy.
    # db, all_hyper_keys = database.load_existing_db(args.tested_model_csv)

    # Lists to hold the results from the *current run*
    total_experiment_summaries = []

    # 2. Scan directories
    # We look for 'algorithm_logs' as the anchor for an experiment dir
    for logs_dir in args.base_dir.rglob('algorithm_logs'):
        experiment_dir = logs_dir.parent
        if not utils.should_select_experiment_dir(experiment_dir.name, args.exp_prefix):
            continue
            
        logging.info("--- Analyzing experiment: %s ---", experiment_dir.relative_to(args.base_dir))

        # 3. Run "heavy" analysis (optional)
        # This includes plotting and re-training, which can be slow.
        if not args.plots:
            analysis.run_per_experiment_analysis(experiment_dir, 
                                                 run_train=(not args.train))
        
        # 4. Parsing (always run in this mode)
        # This reads acc_report.txt and hyper-neural.txt
        network_data_list = parsing.parse_experiment_data(experiment_dir, args.base_dir)
        if not network_data_list:
            logging.warning("No network data found in %s, skipped.", experiment_dir.name)
            continue
        

        # 5. Update DB and create 'total' summary row
        # - db is the global dict of all unique networks
        # - summary_row is just the best network *from this experiment*
        # try:
        summary_row = database.update_model_db(network_data_list)
        # except Exception as e:
        #     logging.error("Failed to update DB for %s: %s", experiment_dir.name, e)
        #     continue
        if summary_row:
            # Add to the list for the *current run's* total.csv
            total_experiment_summaries.append(summary_row)
            
            # --- NEW: Write individual results.csv ---
            try:
                individual_summary_path = experiment_dir / "results.csv"
                database.write_individual_experiment_summary(network_data_list, individual_summary_path)
                logging.info("Wrote individual summary: %s", individual_summary_path.name)
            except Exception as e:
                logging.error("Failed to write individual results.csv for %s: %s", experiment_dir.name, e)
            # --- End New ---
        

    # 6. Final file writing

    # --- File 1: total.csv ---
    # This is the summary of *this run's* best models
    df_total = database.write_total_file(args.total_csv, total_experiment_summaries)
    print(f"✅ Created total summary file: {args.total_csv}")

    # --- File 2: mean.csv ---
    # This aggregates the total.csv from *this run*
    database.write_mean_file(args.mean_csv, df_total)
    print(f"✅ Created mean summary file: {args.mean_csv}")
    

if __name__ == "__main__":
    main()