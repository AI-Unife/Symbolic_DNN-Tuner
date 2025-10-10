#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Aggregate experiment metrics from .out logs into per-experiment CSVs
and a single summary CSV. Designed for directory trees where each
experiment folder contains an output log (`output.log` or .out files)
and metrics lines to extract.

Example:
    python collect_results.py \
        --base-dir /hpc/home/bzzlca/Symbolic_DNN-Tuner/results \
        --exp-prefix 25_

Notes:
- The script walks subfolders under --base-dir and selects those whose
  folder name starts with --exp-prefix (default: "25_").
- For each experiment folder:
  * Collects metrics from all `.out` files into `<exp>/results.csv`
    (one row per iteration hit).
  * Copies a representative log to `<exp>/output/output.log`.
  * Extracts experiment metadata from that log.
  * Summarizes best metrics into a global `total_results.csv`.
"""

from __future__ import annotations

import argparse
import logging
import re
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ---------------------------
# Logging setup
# ---------------------------

def setup_logging(verbosity: int) -> None:
    """Configure logging level and format."""
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


# ---------------------------
# Regex patterns (precompiled)
# ---------------------------

_METRIC_PATTERNS: Dict[str, re.Pattern] = {
    # Accept both ":" and "=" separators; ignore case where it makes sense.
    "iteration": re.compile(r"ITERATION\s*[:=]?\s*(\d+)"),
    "accuracy": re.compile(r"ACCURACY\s*[:=]?\s*([0-9]*\.?[0-9]+)"),
    "flops": re.compile(r"FLOPS\s*[:=]?\s*([0-9]*\.?[0-9]+)"),
    "params": re.compile(r"PARAMS\s*[:=]?\s*([0-9]*\.?[0-9]+)"),
    "latency": re.compile(r"LATENCY\s*[:=]?\s*([0-9]*\.?[0-9]+)"),
    "total_cost": re.compile(r"TOTAL\s*COST\s*[:=]?\s*([0-9]*\.?[0-9]+)"),
    "score": re.compile(r"Score\s*[:=]?\s*(-?[0-9]*\.?[0-9]+)"),
}

# Metadata patterns from `output.log`
_META_PATTERNS: Dict[str, re.Pattern] = {
    "dataset": re.compile(r"DATASET NAME:\s+(.+)", re.IGNORECASE),
    "max_eval": re.compile(r"MAX NET EVAL:\s+(\d+)", re.IGNORECASE),
    "epochs": re.compile(r"EPOCHS FOR TRAINING:\s+(\d+)", re.IGNORECASE),
    "modules": re.compile(r"MODULE LIST:\s+(\[.+\])", re.IGNORECASE),
    "seed": re.compile(r"SEED:\s+(\d+)", re.IGNORECASE),
    "total_time": re.compile(r"TOTAL TIME -------->\s*([0-9]*\.?[0-9]+)", re.IGNORECASE),
}

# ---------------------------
# Search space evolution (per-iteration snapshot + diffs)
# ---------------------------

_ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")

def _strip_ansi(s: str) -> str:
    """Remove ANSI color codes from a log line."""
    return _ANSI_RE.sub("", s)

def _iter_search_space_blocks(lines: List[str]):
    """
    Generator over (iteration, block_lines) where block_lines are the lines
    immediately following 'Actual search space:' for that iteration.
    """
    iter_no = None
    i = 0
    n = len(lines)
    while i < n:
        line = _strip_ansi(lines[i])
        m_iter = re.search(r"\bITERATION\s+(\d+)\b", line)
        if m_iter:
            iter_no = int(m_iter.group(1))
        if "Actual search space:" in line and iter_no is not None:
            # collect subsequent 'Dimension ...' lines until a non-dimension line
            block = []
            j = i + 1
            while j < n:
                l = _strip_ansi(lines[j]).strip()
                if not l.startswith("Dimension "):
                    break
                block.append(l)
                j += 1
            yield iter_no, block
            i = j
            continue
        i += 1

def _parse_dimension_line(line: str):
    """
    Parse a single 'Dimension k: name - TYPE(low=..., high=..., ...)' line.
    Returns (index:int, name:str, kind:str, low:float|None, high:float|None, raw:str)
    Works for Integer/Real; for Categorical falls back with None bounds.
    """
    # Example:
    # Dimension 0: unit_c1 - Integer(low=32, high=64, prior='uniform', transform='identity')
    m = re.match(r"Dimension\s+(\d+):\s*([^-\s]+)\s*-\s*([A-Za-z]+)\((.*)\)$", line)
    if not m:
        return None
    idx = int(m.group(1))
    name = m.group(2)
    kind = m.group(3)
    inner = m.group(4)

    low = high = None
    if kind.lower() in ("integer", "real"):
        m2 = re.search(r"low\s*=\s*([0-9]*\.?[0-9]+)", inner)
        m3 = re.search(r"high\s*=\s*([0-9]*\.?[0-9]+)", inner)
        if m2:
            low = float(m2.group(1))
        if m3:
            high = float(m3.group(1))
    return idx, name, kind, low, high

def extract_search_space_evolution(log_file: Path) -> pd.DataFrame:
    """
    Build a per-iteration table of the search space and how it changes over time.
    Columns:
        iteration, dim_index, param, kind, low, high,
        prev_low, prev_high, status, delta_low, delta_high, raw
    - status ∈ {'initial','unchanged','changed'}
    """
    try:
        with log_file.open("r", encoding="utf-8", errors="ignore") as fh:
            lines = fh.readlines()
    except FileNotFoundError:
        logging.warning("Log file not found for search space evolution: %s", log_file)
        return pd.DataFrame()

    snapshots: List[Dict[str, object]] = []
    for it, block in _iter_search_space_blocks(lines):
        for ln in block:
            parsed = _parse_dimension_line(ln)
            if not parsed:
                continue
            dim_idx, name, kind, low, high = parsed
            snapshots.append({
                "iteration": it,
                "dim_index": dim_idx,
                "param": name,
                "kind": kind,
                "low": low,
                "high": high
            })

    if not snapshots:
        logging.warning("No 'Actual search space' blocks found in %s", log_file)
        return pd.DataFrame()

    df = pd.DataFrame(snapshots)
    # Normalize dtypes
    df = df.sort_values(["iteration", "dim_index"]).reset_index(drop=True)

    for idx, row in df.iterrows():
        if row.kind in ['Real', 'Integer']:
            df.at[idx, 'width'] = df.at[idx, 'high'] - df.at[idx,'low']
        elif row.kind in ['Categorical']:
            df.at[idx, 'width'] = np.nan
    return df


def write_search_space_evolution(exp_dir: Path, log_file: Optional[Path]) -> Optional[Path]:
    """
    Create `<exp_dir>/search_space_evolution.csv` by parsing the representative log.
    Returns the CSV path if written, else None.
    """
    if not log_file or not log_file.is_file():
        logging.warning("No representative log available to build search space evolution for %s", exp_dir)
        return None
    df = extract_search_space_evolution(log_file)
    if df.empty:
        return None
    out_path = exp_dir / "search_space_evolution.csv"
    df.to_csv(out_path, index=False)
    logging.info("Wrote search space evolution CSV: %s", out_path)
    return out_path

def build_stacked_fractional_change_plot(csv_path: Path,
                                         out_png: Path,
                                         out_csv: Path | None = None) -> Path:
    """
    From a CSV (iteration,param,low,high), compute fractional deltas per iteration:
      - New dimension at iteration t  -> delta = 1.0
      - Existing: (w_t - w_{t-1}) / w_{t-1}
      - If w_{t-1} == 0 and w_t > 0   -> delta = 1.0
      - If w_{t-1} == 0 and w_t == 0  -> delta = 0.0
      - Missing/disappeared param     -> delta = 0.0
    Then, plot a stacked bar chart (one bar per iteration; segments=dimensions).
    Optionally save the deltas table to out_csv.
    """
    df = pd.read_csv(csv_path)

    def _resolve(colname: str) -> str:
        low = colname.lower()
        for c in df.columns:
            if c.lower() == low:
                return c
        raise ValueError(f"Missing column '{colname}' in {list(df.columns)}")

    col_iter = _resolve("iteration")
    try:
        col_param = _resolve("param")
    except ValueError:
        try:
            col_param = _resolve("name")
        except ValueError:
            col_param = _resolve("dimension")
    col_width = _resolve("width")

    work = df[[col_iter, col_param, col_width]].copy()
    work[col_iter] = pd.to_numeric(work[col_iter], errors="coerce")
    work[col_width] = pd.to_numeric(work[col_width], errors="coerce")
    work = work.dropna(subset=[col_iter, col_width])

    widths = work.pivot_table(index=col_iter, columns=col_param, values="width", aggfunc="first").sort_index()
    iters = widths.index.tolist()
    params = sorted(set(widths.columns))

    deltas = pd.DataFrame(0.0, index=iters, columns=params)
    prev_row = None
    for i, it in enumerate(iters):
        row = widths.loc[it]
        if i == 0:
            for p in params:
                deltas.at[it, p] = 1.0 if pd.notna(row.get(p)) else 0.0
        else:
            for p in params:
                cur = row.get(p) if p in row.index else np.nan
                prev_exists = prev_row is not None and (p in prev_row.index) and pd.notna(prev_row[p])
                if pd.isna(cur) and not prev_exists:
                    deltas.at[it, p] = 0.0
                elif pd.isna(cur) and prev_exists:
                    deltas.at[it, p] = -1.0
                elif not prev_exists:
                    deltas.at[it, p] = 1.0
                else:
                    prev_w = prev_row[p]
                    cur_w = cur
                    if prev_w == 0:
                        deltas.at[it, p] = 1.0 if cur_w > 0 else 0.0
                    else:
                        deltas.at[it, p] = (cur_w - prev_w) / prev_w
                        # if p == 'dr_f':
                        #     print("iter {} - delta {} - cur_w {} - prev_w {}".format(it, deltas.at[it, p], cur_w, prev_w))
        prev_row = row

    if out_csv:
        deltas.to_csv(out_csv)

    # Plot (one figure, default colors, stacked)
    deltas = deltas[1:]
    x = np.arange(len(deltas.index))
    bottom = np.zeros(len(deltas), dtype=float)
    plt.figure(figsize=(12, 6))
    ax = plt.gca()
    for p in deltas.columns:
        vals = deltas[p].fillna(0.0).values
        ax.bar(x, vals, bottom=bottom, label=str(p))
        bottom = bottom + vals
    ax.set_xticks(x)
    try:
        ax.set_xticklabels([int(v) for v in deltas.index], rotation=0)
    except Exception:
        ax.set_xticklabels(list(deltas.index), rotation=0)
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Fractional Δ of width (1 = +100%)")
    ax.set_title("Stacked fractional change of search-space dimensions per iteration")
    ax.legend(loc="best", ncols=2)
    plt.tight_layout()
    plt.savefig(out_png, dpi=200, bbox_inches="tight")
    plt.close()
    return out_png

# ---------------------------
# Data structures
# ---------------------------

@dataclass
class ExperimentInfo:
    """Container for experiment-level metadata and best metrics."""
    name: str
    tuner: Optional[str] = None
    dataset_name: str = ""
    max_net_eval: float = np.nan
    epochs_for_training: float = np.nan
    seed: float = np.nan
    total_time_s: float = 86400000  # default to 1 day if absent
    modules_flags: Dict[str, bool] = field(default_factory=dict)
    best_iteration: Optional[int] = None
    eval_count: int = 0
    # Best numeric metrics (e.g., accuracy, flops, ...)
    best_metrics: Dict[str, float] = field(default_factory=dict)


# ---------------------------
# Parsing utils
# ---------------------------

def extract_tuner(exp_name: str) -> Optional[str]:
    """
    Heuristically infer the tuner name from the experiment folder string.
    Expand this list if you add more tuners.
    """
    tuner_names = ["filtered", "basic", "standard", "RS_ruled", "RS"]
    for t in tuner_names:
        if t in exp_name:
            return t
    # Additional heuristic: look for tokens after position 6 like original code
    parts = exp_name.split("_")
    if len(parts) > 7:
        cand = parts[6]
        if parts[7] == "ruled":
            cand += "_ruled"
        return cand
    return None


def parse_modules_list(raw: Optional[str], all_modules: Iterable[str]) -> Dict[str, bool]:
    """
    Robustly detect which modules are present, even if the meta line is malformed.
    """
    flags = {m: False for m in all_modules}
    if not raw:
        return flags
    # Lower both sides and do substring inclusion
    raw_low = raw.lower()
    for m in all_modules:
        flags[m] = m.lower() in raw_low
    return flags


def parse_experiment_meta(log_text: str, all_modules: Iterable[str]) -> Tuple[str, Dict[str, float | str], Dict[str, bool]]:
    """
    Extract metadata fields from a full log text using precompiled regexes.
    Returns (dataset_name, numeric_fields, modules_flags).
    """
    dataset = _META_PATTERNS["dataset"].search(log_text)
    max_eval = _META_PATTERNS["max_eval"].search(log_text)
    epochs = _META_PATTERNS["epochs"].search(log_text)
    modules = _META_PATTERNS["modules"].search(log_text)
    seed = _META_PATTERNS["seed"].search(log_text)
    total_time = _META_PATTERNS["total_time"].search(log_text)

    numeric_fields: Dict[str, float | str] = {
        "Max Net Eval": float(max_eval.group(1)) if max_eval else np.nan,
        "Epochs for Training": float(epochs.group(1)) if epochs else np.nan,
        "seed": float(seed.group(1)) if seed else np.nan,
        "Total Time (s)": float(total_time.group(1)) if total_time else 86400.0,
    }
    modules_flags = parse_modules_list(modules.group(1).strip() if modules else "", all_modules)
    dataset_name = dataset.group(1).strip() if dataset else ""
    return dataset_name, numeric_fields, modules_flags


# ---------------------------
# Metrics extraction from .out
# ---------------------------

def extract_metrics_from_out(file_path: Path) -> pd.DataFrame:
    """
    Parse a single `.out` file and return a DataFrame of metric rows.
    A row is finalized when we see an iteration boundary (a new "ITERATION" line).
    """
    rows: List[Dict[str, float | int | None]] = []
    current = {k: None for k in _METRIC_PATTERNS.keys()}

    try:
        with file_path.open("r", encoding="utf-8", errors="ignore") as fh:
            for line in fh:
                # Accumulate known metrics from the current line
                for key, pat in _METRIC_PATTERNS.items():
                    m = pat.search(line)
                    if m:
                        if key == "iteration":
                            current[key] = int(m.group(1))
                        else:
                            current[key] = float(m.group(1))
                # Use the *appearance* of ITERATION as a boundary to finalize a row
                if _METRIC_PATTERNS["iteration"].search(line):
                    # Only append if we have iteration and at least one other metric (e.g., score)
                    if current["iteration"] is not None:
                        rows.append(current.copy())
                    # Start a fresh row for the next iteration block
                    current = {k: None for k in _METRIC_PATTERNS.keys()}
    except FileNotFoundError:
        logging.warning("File not found: %s", file_path)
        return pd.DataFrame()
    except Exception as e:
        logging.exception("Failed to parse %s: %s", file_path, e)
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    # Drop rows with no iteration number (shouldn't happen, but be safe)
    if "iteration" in df.columns:
        df = df.dropna(subset=["iteration"]).reset_index(drop=True)
        # Ensure dtypes are correct
        df["iteration"] = df["iteration"].astype(int)
    return df


def process_out_files_in_dir(exp_dir: Path) -> Optional[Path]:
    """
    Walk an experiment directory and parse *all* `.out` files found.
    Concatenate their metric rows and write to `<exp_dir>/results.csv`.
    Returns the path to the CSV, or None if nothing was written.
    """
    all_rows: List[pd.DataFrame] = []
    for path in exp_dir.rglob("*.out"):
        logging.info("Parsing: %s", path.name)
        df = extract_metrics_from_out(path)
        if not df.empty:
            all_rows.append(df)

    if not all_rows:
        logging.warning("No .out metrics found under: %s", exp_dir)
        return None

    results = pd.concat(all_rows, ignore_index=True).sort_values("iteration").reset_index(drop=True)
    csv_path = exp_dir / "results.csv"
    results.to_csv(csv_path, index=False)
    logging.info("Wrote metrics CSV: %s", csv_path)
    return csv_path


# ---------------------------
# Log file management
# ---------------------------

def copy_log_to_output(log_src: Path, exp_dir: Path) -> Optional[Path]:
    """
    Copy a representative log (e.g., the largest .out or an explicit file)
    to `<exp_dir>/output/output.log` for easy inspection.
    Returns destination path or None if operation failed.
    """
    dest_dir = exp_dir / "output"
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / "output.log"

    try:
        shutil.copy2(log_src, dest)
        logging.info("Copied log: %s -> %s", log_src, dest)
        return dest
    except Exception as e:
        logging.warning("Failed to copy %s to %s: %s", log_src, dest, e)
        return None


def pick_representative_log(exp_dir: Path) -> Optional[Path]:
    """
    Choose a representative log to copy:
    - Prefer 'output.log' if it already exists.
    - Else pick the largest `.out` file (often the most complete).
    """
    explicit = exp_dir / "output.log"
    if explicit.is_file():
        return explicit

    outs = list(exp_dir.rglob("*.out"))
    if not outs:
        return None
    # Pick the largest by size
    outs.sort(key=lambda p: p.stat().st_size if p.exists() else 0, reverse=True)
    return outs[0]


# ---------------------------
# Experiment scanning
# ---------------------------

def should_select_experiment_dir(dir_name: str, prefix: str) -> bool:
    """Return True if the folder should be considered an experiment."""
    return dir_name.startswith(prefix)


def summarize_experiment(exp_dir: Path, all_modules: Iterable[str]) -> Optional[ExperimentInfo]:
    """
    Create an ExperimentInfo from <exp_dir>:
    - Ensure metrics CSV exists (process .out files as needed)
    - Copy representative log to output/output.log
    - Parse metadata and best metrics
    """
    exp_name = exp_dir.name
    logging.info("Processing experiment: %s", exp_name)

    # 1) Ensure results.csv exists (aggregate from .out files)
    results_csv = exp_dir / "results.csv"
    if not results_csv.is_file():
        results_csv = process_out_files_in_dir(exp_dir)  # may be None

    # 2) Ensure we have a representative log saved at output/output.log
    rep_log = pick_representative_log(exp_dir)
    output_log = None
    if rep_log:
        output_log = copy_log_to_output(rep_log, exp_dir)

    # If both are missing, there's nothing to summarize
    if not results_csv or not results_csv.is_file():
        logging.warning("Missing results.csv for %s; skipping.", exp_name)
        return None

    # 3) Initialize experiment info
    info = ExperimentInfo(name=exp_name)
    info.tuner = extract_tuner(exp_name)

    # 4) Parse log metadata if we have a log
    if output_log and output_log.is_file():
        # Build per-iteration search space evolution CSV
        try:
            write_search_space_evolution(exp_dir, output_log)
        except Exception as e:
            logging.warning("Failed to write search space evolution for %s: %s", exp_name, e)
        csv_evo = exp_dir / "search_space_evolution.csv"
        if csv_evo.exists():
            try:
                build_stacked_fractional_change_plot(
                    csv_path=csv_evo,
                    out_png=exp_dir / "search_space_pct_changes.png",
                    out_csv=exp_dir / "search_space_pct_changes.csv"
                )
            except Exception as e:
                logging.warning("Unable to plot stacked %% changes for %s: %s", exp_dir.name, e)
        try:
            text = output_log.read_text(encoding="utf-8", errors="ignore")
            dataset_name, numeric_fields, modules_flags = parse_experiment_meta(text, all_modules)
            info.dataset_name = dataset_name
            info.max_net_eval = numeric_fields["Max Net Eval"]
            info.epochs_for_training = numeric_fields["Epochs for Training"]
            info.seed = numeric_fields["seed"]
            info.total_time_s = numeric_fields["Total Time (s)"]
            info.modules_flags = modules_flags
        except Exception as e:
            logging.warning("Failed to parse metadata for %s: %s", exp_name, e)

    # 5) Load results and compute best metrics
    try:
        df = pd.read_csv(results_csv)
        if df.empty:
            logging.warning("Empty results for %s; skipping.", exp_name)
            return None

        # Ensure we have numeric columns identified
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

        # Best iteration by highest accuracy (if available)
        if "score" in df.columns and not df["score"].dropna().empty:
            best_idx = int(df["score"].idxmax())
            info.best_iteration = int(df.loc[best_idx, "iteration"]) if "iteration" in df.columns else None
        else:
            info.best_iteration = None

        # For each numeric column, select the max (typical for accuracy/score).
        # If you need min for "latency" or "total_cost", adjust as needed.
        for col in numeric_cols:
            if col == "iteration":
                continue
            series = df[col].dropna()
            if col in ["latency", "total_cost", "flops", "score"]:
                info.best_metrics[f"Best {col}"] = float(series.min()) if not series.empty else np.nan
            else:
                info.best_metrics[f"Best {col}"] = float(series.max()) if not series.empty else np.nan

        info.eval_count = int(len(df["accuracy"])) if "accuracy" in df.columns else int(len(df))
    except pd.errors.EmptyDataError:
        logging.warning("Empty CSV for %s; skipping.", exp_name)
        return None
    except Exception as e:
        logging.exception("Failed summarizing results for %s: %s", exp_name, e)
        return None

    return info


# ---------------------------
# Main pipeline
# ---------------------------

def run(
    base_dir: Path,
    exp_prefix: str = "25_",
    all_modules: Optional[List[str]] = None,
    output_csv: Path = Path("total_results.csv"),
) -> Path:
    """
    Walk the base directory, summarize each experiment, and write a global CSV.
    """
    if all_modules is None:
        all_modules = ["accuracy_module", "flops_module", "hardware_module"]

    exp_infos: List[ExperimentInfo] = []

    # Only consider immediate children of base_dir (as in original code),
    # but you can switch to rglob if you want deeper scanning.
    for child in sorted(base_dir.iterdir()):
        if not child.is_dir():
            continue
        if not should_select_experiment_dir(child.name, exp_prefix):
            continue

        logging.info("Selected experiment folder: %s", child.name)
        info = summarize_experiment(child, all_modules)
        if info:
            exp_infos.append(info)

    if not exp_infos:
        logging.warning("No experiments summarized. Nothing to write.")
        # Still create an empty CSV for consistency
        pd.DataFrame().to_csv(output_csv, index=False)
        pd.DataFrame().to_excel(output_csv.with_suffix('.xlsx'), index=False)
        return output_csv

    # Convert to a flat DataFrame
    rows: List[Dict[str, object]] = []
    for info in exp_infos:
        base_row: Dict[str, object] = {
            "Experiment Name": info.name,
            "Tuner": info.tuner or "",
            "Dataset Name": info.dataset_name,
            "Max Net Eval": info.max_net_eval,
            "Epochs for Training": info.epochs_for_training,
            "seed": info.seed,
            "Total Time (s)": info.total_time_s,
            "Best Iteration": info.best_iteration if info.best_iteration is not None else np.nan,
            "Eval": info.eval_count,
        }
        # Module flags
        for m in (all_modules or []):
            base_row[m] = bool(info.modules_flags.get(m, False))
        # Best metrics
        base_row.update(info.best_metrics)
        rows.append(base_row)

    df_total = pd.DataFrame(rows)
    df_total.to_csv(output_csv, index=False)
    df_total.to_excel(output_csv.with_suffix('.xlsx'), index=False)
    logging.info("Wrote summary CSV: %s", output_csv)
    return output_csv


# ---------------------------
# CLI
# ---------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Collect and summarize metrics from experiment .out logs."
    )
    parser.add_argument(
        "--base-dir",
        type=Path,
        default=Path("/Users/alicebizzarri/Desktop/Work/AIDA4Edge/tuner-insight-dashboard/public/experiments/resultsFLOPS"),
        help="Folder that contains experiment subfolders.",
    )
    parser.add_argument(
        "--exp-prefix",
        type=str,
        default="25_",
        help="Only process experiment folders whose names start with this prefix.",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path("total_results.csv"),
        help="Path to write the global summary CSV.",
    )
    parser.add_argument(
        "--modules",
        type=str,
        nargs="*",
        default=["accuracy_module", "flops_module", "hardware_module"],
        help="Module names to detect in the logs.",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="count",
        default=0,
        help="Increase verbosity (-v for INFO, -vv for DEBUG).",
    )
    parser.add_argument(
        "--only-metrics",
        type=bool,
        default=False,
        action=argparse.BooleanOptionalAction,
        help="Only print mean best metrics by tuner from total_results.csv and exit.",
    )
    return parser.parse_args()

import pandas as pd
from typing import Optional

def print_mean_best_by_tuner(csv_path: str, out_csv: Optional[str] = None) -> pd.DataFrame:
    """
    Legge `total_results.csv`, raggruppa per Tuner e calcola la media (± std)
    delle colonne che iniziano con 'Best' e di 'Total Time (s)' (convertito in ore).

    Parametri
    ---------
    csv_path : str
        Path al total_results.csv.
    out_csv : Optional[str]
        Se fornito, salva il risultato anche su CSV.

    Ritorna
    -------
    pd.DataFrame
        DataFrame indicizzato per Tuner con media e std per ogni metrica.
    """
    df = pd.read_csv(csv_path)
    if "Tuner" not in df.columns:
        raise ValueError("Nel CSV manca la colonna 'Tuner'.")

    # Seleziona colonne Best + tempo
    best_cols = [c for c in df.columns if c.lower().startswith("best")]
    time_col = "Total Time (s)" if "Total Time (s)" in df.columns else None

    if not best_cols and not time_col:
        raise ValueError("Nessuna colonna 'Best*' o 'Total Time (s)' trovata nel CSV.")

    # Prepara dataframe di lavoro
    work = df[["Tuner"]].copy()
    for c in best_cols:
        work[c] = pd.to_numeric(df[c], errors="coerce")

    if time_col:
        # Converti secondi in ore
        work["Total Time (h)"] = pd.to_numeric(df[time_col], errors="coerce") / 3600.0

    # Calcola media e std per tuner
    grouped = work.groupby("Tuner", dropna=False)
    mean_df = grouped.mean(numeric_only=True)
    std_df = grouped.std(numeric_only=True)

    # Crea un DataFrame con media ± std (formattato)
    formatted_df = pd.DataFrame(index=mean_df.index, columns=mean_df.columns)
    for col in mean_df.columns:
        for tuner in mean_df.index:
            m = mean_df.loc[tuner, col]
            s = std_df.loc[tuner, col]
            if pd.isna(m):
                formatted_df.loc[tuner, col] = "-"
            else:
                formatted_df.loc[tuner, col] = f"{m:.4f} (±{s:.4f})"

    # Stampa leggibile
    print("\n================  MEAN (±STD) OF 'Best*' METRICS BY TUNER  ================\n")
    for tuner in mean_df.index:
        print(f"▶ Tuner: {tuner}")
        for col in mean_df.columns:
            val = formatted_df.loc[tuner, col]
            print(f"  - {col}: {val}")
        print()

    # Salva CSV opzionale con i valori numerici (non formattati)
    if out_csv:
        mean_out = mean_df.copy()
        if time_col:
            mean_out.rename(columns={"Total Time (h)": "Mean Total Time (h)"}, inplace=True)
            std_df.rename(columns={"Total Time (h)": "Std Total Time (h)"}, inplace=True)
        combined = mean_out.add_suffix("_mean").join(std_df.add_suffix("_std"))
        combined.to_csv(out_csv, index=True)
        print(f"📁 Salvato file: {out_csv}")

    return formatted_df

def main() -> None:
    args = parse_args()
    setup_logging(args.verbose)
    logging.info("Base dir: %s", args.base_dir)
    if not args.only_metrics:
        run(
            base_dir=args.base_dir,
            exp_prefix=args.exp_prefix,
            all_modules=args.modules,
            output_csv=args.output_csv,
        )
        print("✅ Created:", args.output_csv)
    print_mean_best_by_tuner("total_results.csv")


if __name__ == "__main__":
    main()
