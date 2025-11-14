#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Main control panel for experiment analysis.

This script orchestrates the parsing, database, and analysis modules
to generate the requested CSV reports.
"""

import argparse
import logging
from pathlib import Path

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
    
    # Verbosity flag
    parser.add_argument("-v", "--verbose", action="count", default=0,
                        help="Increase verbosity (-v for INFO, -vv for DEBUG).")
    return parser.parse_args()

def main():
    args = parse_args()
    setup_logging(args.verbose)
    
    logging.info("Starting analysis, Base dir: %s", args.base_dir)

    # 1. Load the existing 'tested_model.csv' database
    db, all_hyper_keys = database.load_existing_db(args.tested_model_csv)

    # Lists to hold the results
    total_experiment_summaries = []
    total_networks_scanned = 0

    # 2. Scan directories
    for logs_dir in args.base_dir.rglob('algorithm_logs'):
        experiment_dir = logs_dir.parent
        if not utils.should_select_experiment_dir(experiment_dir.name, args.exp_prefix):
            continue
            
        logging.info("--- Analyzing experiment: %s ---", experiment_dir.relative_to(args.base_dir))

        # 3. Run "heavy" analysis (optional)
        if not args.plots:
            analysis.run_per_experiment_analysis(experiment_dir, 
                                                 run_train=(not args.train))
        
        # 4. Parsing (always run)
        network_data_list = parsing.parse_experiment_data(experiment_dir, args.base_dir)
        if not network_data_list:
            logging.warning("No network data found in %s, skipped.", experiment_dir.name)
            continue
        
        total_networks_scanned += len(network_data_list)

        # 5. Update DB and create 'total' summary row
        summary_row, updated_count, new_count = database.update_model_db(db, network_data_list, all_hyper_keys)
        if summary_row:
            total_experiment_summaries.append(summary_row)
            
        if updated_count or new_count:
             logging.info("Merging 'tested_model.csv': %d new, %d updated from %s",
                          new_count, updated_count, experiment_dir.name)

    # 6. Final file writing
    
    # --- File 1: tested_model.csv ---
    db_records = list(db.values())
    database.write_tested_model_file(args.tested_model_csv, db_records, all_hyper_keys)
    print(f"✅ Created tested models file: {args.tested_model_csv}")

    # --- File 2: total.csv ---
    df_total = database.write_total_file(args.total_csv, total_experiment_summaries)
    print(f"✅ Created total summary file: {args.total_csv}")

    # --- File 3: mean.csv ---
    database.write_mean_file(args.mean_csv, df_total)
    print(f"✅ Created mean summary file: {args.mean_csv}")
    
    # --- Console Summary ---
    print("\n--- Execution Summary ---")
    print(f"Total networks scanned in log files (current run): {total_networks_scanned}")
    print(f"Unique networks (total, after merge and filter):    {len(db_records)}")

if __name__ == "__main__":
    main()