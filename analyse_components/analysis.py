# In analysis.py
import logging
import re
import shutil
from pathlib import Path
# FIX: Import 'Optional' for Python < 3.10 compatibility
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Import TF modules only if available
try:
    import tensorflow as tf
    from components.dataset import get_datasets
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False
    
# Import utility and parsing functions
from analyse_components import utils
from analyse_components import parsing # For get_best_iteration_from_acc, parse_batch...

# --- Retrain Network (Full) ---
def get_best_iteration_from_acc(aldir: Path) -> Optional[int]:
    """Returns the 1-based index of the row with max accuracy in acc_report.txt."""
    acc_path = aldir / "acc_report.txt"
    if not acc_path.exists(): return None
    vals = []
    for ln in acc_path.read_text().splitlines():
        ln = ln.strip();
        if not ln: continue
        try: vals.append(float(ln))
        except ValueError: pass
    if not vals: return None
    best_idx = int(pd.Series(vals).idxmax())
    return best_idx + 1

def parse_batch_opt_lr_from_hyper_neural(aldir: Path, iteration: int) -> Optional[str]:
    """Gets the (batch_size, optimizer, learning_rate) for a specific 1-based iteration."""
    f = aldir / "hyper-neural.txt"
    if not f.exists(): return None
    rows = []
    for ln in f.read_text().splitlines():
        ln = ln.strip();
        if not ln: continue
        try:
            d = ast.literal_eval(ln)
            if isinstance(d, dict): rows.append(d)
        except Exception: continue
    if not rows: return None
    idx = max(0, min(iteration - 1, len(rows) - 1))
    batch = rows[idx].get("batch_size")
    opt = rows[idx].get("optimizer")
    lr = rows[idx].get("learning_rate")
    return (batch, opt, lr)
    
def load_dataset(dataset_name: str):
    """Helper to load dataset using user's components."""
    x_train, y_train, x_test, y_test, n_class = get_datasets(dataset_name.lower().replace("-", " "))
    return (x_train, y_train), (x_test, y_test)

def make_optimizer_by_name(name: str, lr: float = 1e-3):
    """Helper to instantiate a Keras optimizer by its string name."""
    name_low = (name or "").lower()
    if name_low == "adamw":
        try: return tf.keras.optimizers.AdamW(learning_rate=lr, weight_decay=0.0)
        except Exception: return tf.keras.optimizers.Adam(learning_rate=lr)
    if name_low == "adamax": return tf.keras.optimizers.Adamax(learning_rate=lr)
    if name_low == "adam": return tf.keras.optimizers.Adam(learning_rate=lr)
    if name_low == "sgd": return tf.keras.optimizers.SGD(learning_rate=lr, momentum=0.9, nesterov=True)
    if name_low == "rmsprop": return tf.keras.optimizers.RMSprop(learning_rate=lr, momentum=0.9)
    return tf.keras.optimizers.Adam(learning_rate=lr)
    
def train_best_model_if_required(exp_dir: Path):
    """
    Main function to load, compile, and re-train the best model
    found during the search phase.
    """
    if not TF_AVAILABLE:
        logging.warning("Training requested, but TensorFlow is not available. Skipping.")
        return None, None
        
    model_path = exp_dir / "Model" / "best-model.keras"
    try:
        dataset_name = exp_dir.name.split("_")[5]
    except IndexError:
        logging.error("Could not extract dataset name from %s", exp_dir.name)
        return None, None
        
    aldir = exp_dir / "algorithm_logs" 
    if not model_path.exists():
        logging.warning("best-model.keras not found in %s", model_path)
        return None, None
        
    model = tf.keras.models.load_model(model_path)
    best_it = get_best_iteration_from_acc(aldir)
    
    if not best_it:
        logging.warning("Could not determine best iteration from acc_report.txt")
        return None, None
        
    (batch, opt_name, lr) = parse_batch_opt_lr_from_hyper_neural(aldir, best_it) or (None, None, None)
    opt_name = opt_name or "Adam"
    lr = lr or 1e-3
    batch = batch or 32
    
    train_split, val_split = load_dataset(dataset_name)
    optimizer = make_optimizer_by_name(opt_name, lr=lr)
    loss = tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.0)
    metrics = [tf.keras.metrics.CategoricalAccuracy(name="acc")]
    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
    
    callbacks = [tf.keras.callbacks.EarlyStopping(monitor="val_acc", mode="max", patience=30, restore_best_weights=True)]
    history = None
    
    logging.info("Starting 'refit' for %s (optimizer=%s, lr=%.3g, batch=%d)", exp_dir.name, opt_name, lr, batch)
    if isinstance(train_split, tuple):
        (x_tr, y_tr) = train_split
        (x_va, y_va) = val_split
        history = model.fit(x_tr, y_tr, validation_data=(x_va, y_va), epochs=1000, batch_size=batch, callbacks=callbacks, verbose=2)
    else:
        history = model.fit(train_split, validation_data=val_split, epochs=1000, callbacks=callbacks, verbose=2)
        
    finetuned_path = exp_dir / "Model" / "best-model-finetuned.keras"
    try: model.save(finetuned_path)
    except Exception as e: logging.warning("Failed to save fine-tuned model: %s", e)
    
    if history is not None:
        hist_df = pd.DataFrame(history.history)
        hist_csv = exp_dir / "best_model_finetune_history.csv"
        hist_df.to_csv(hist_csv, index=False)
        
    logging.info("Best model training complete.")
    acc_test, loss_test = model.evaluate(val_split, verbose=2) if not isinstance(val_split, tuple) else model.evaluate(x_va, y_va, verbose=2)
    return acc_test, loss_test

# --- Search Space (Full) ---
_ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")

def _strip_ansi(s: str) -> str: return _ANSI_RE.sub("", s)

def _iter_search_space_blocks(lines: List[str]):
    """Generator for (iteration, block_lines) from 'Actual search space:' blocks."""
    iter_no = None; i = 0; n = len(lines)
    while i < n:
        line = _strip_ansi(lines[i])
        m_iter = re.search(r"\bITERATION\s+(\d+)\b", line)
        if m_iter: iter_no = int(m_iter.group(1))
        if "Actual search space:" in line and iter_no is not None:
            block = []
            j = i + 1
            while j < n:
                l = _strip_ansi(lines[j]).strip()
                if not l.startswith("Dimension "): break
                block.append(l); j += 1
            yield iter_no, block
            i = j; continue
        i += 1

def _parse_dimension_line(line: str):
    """Parses a single 'Dimension k: name - TYPE(...)' line."""
    m = re.match(r"Dimension\s+(\d+):\s*([^-\s]+)\s*-\s*([A-Za-z]+)\((.*)\)$", line)
    if not m: return None
    idx = int(m.group(1)); name = m.group(2); kind = m.group(3); inner = m.group(4)
    low = high = None
    if kind.lower() in ("integer", "real"):
        m2 = re.search(r"low\s*=\s*([0-9]*\.?[0-9]+)", inner)
        m3 = re.search(r"high\s*=\s*([0-9]*\.?[0-9]+)", inner)
        if m2: low = float(m2.group(1))
        if m3: high = float(m3.group(1))
    return idx, name, kind, low, high

def extract_search_space_evolution(log_file: Path) -> pd.DataFrame:
    """Builds a per-iteration table of the search space dimensions."""
    try:
        with log_file.open("r", encoding="utf-8", errors="ignore") as fh: lines = fh.readlines()
    except FileNotFoundError:
        logging.warning("Log file not found for search space evolution: %s", log_file)
        return pd.DataFrame()
    snapshots: List[Dict[str, object]] = []
    for it, block in _iter_search_space_blocks(lines):
        for ln in block:
            parsed = _parse_dimension_line(ln)
            if not parsed: continue
            dim_idx, name, kind, low, high = parsed
            snapshots.append({"iteration": it, "dim_index": dim_idx, "param": name, "kind": kind, "low": low, "high": high})
    if not snapshots:
        logging.warning("No 'Actual search space' blocks found in %s", log_file)
        return pd.DataFrame()
    df = pd.DataFrame(snapshots).sort_values(["iteration", "dim_index"]).reset_index(drop=True)
    for idx, row in df.iterrows():
        if row.kind in ['Real', 'Integer']:
            df.at[idx, 'width'] = df.at[idx, 'high'] - df.at[idx,'low']
        elif row.kind in ['Categorical']:
            df.at[idx, 'width'] = np.nan
    return df

def write_search_space_evolution(exp_dir: Path, log_file: Optional[Path]) -> Optional[Path]:
    """Saves the search space evolution DataFrame to a CSV."""
    if not log_file or not log_file.is_file():
        logging.warning("No representative log available for search space evolution: %s", exp_dir)
        return None
    df = extract_search_space_evolution(log_file)
    if df.empty: return None
    out_path = exp_dir / "search_space_evolution.csv"
    df.to_csv(out_path, index=False)
    logging.info("Wrote search space evolution CSV: %s", out_path)
    return out_path

#
# --- FIX APPLIED HERE ---
#
# Replaced 'Path | None' with 'Optional[Path]' for Python < 3.10
#
def build_stacked_fractional_change_plot(csv_path: Path, out_png: Path, out_csv: Optional[Path] = None) -> Path:
    """Generates a stacked bar chart of the fractional change in dimension widths."""
    df = pd.read_csv(csv_path)
    def _resolve(colname: str) -> str:
        low = colname.lower()
        for c in df.columns:
            if c.lower() == low: return c
        raise ValueError(f"Missing column '{colname}' in {list(df.columns)}")
    col_iter = _resolve("iteration"); col_param = _resolve("param"); col_width = _resolve("width")
    work = df[[col_iter, col_param, col_width]].copy()
    work[col_iter] = pd.to_numeric(work[col_iter], errors="coerce")
    work[col_width] = pd.to_numeric(work[col_width], errors="coerce")
    work = work.dropna(subset=[col_iter, col_width])
    widths = work.pivot_table(index=col_iter, columns=col_param, values="width", aggfunc="first").sort_index()
    iters = widths.index.tolist(); params = sorted(set(widths.columns))
    deltas = pd.DataFrame(0.0, index=iters, columns=params)
    prev_row = None
    for i, it in enumerate(iters):
        row = widths.loc[it]
        if i == 0:
            for p in params: deltas.at[it, p] = 1.0 if pd.notna(row.get(p)) else 0.0
        else:
            for p in params:
                cur = row.get(p) if p in row.index else np.nan
                prev_exists = prev_row is not None and (p in prev_row.index) and pd.notna(prev_row[p])
                if pd.isna(cur) and not prev_exists: deltas.at[it, p] = 0.0
                elif pd.isna(cur) and prev_exists: deltas.at[it, p] = -1.0
                elif not prev_exists: deltas.at[it, p] = 1.0
                else:
                    prev_w = prev_row[p]; cur_w = cur
                    if prev_w == 0: deltas.at[it, p] = 1.0 if cur_w > 0 else 0.0
                    else: deltas.at[it, p] = (cur_w - prev_w) / prev_w
        prev_row = row
    if out_csv: deltas.to_csv(out_csv)
    deltas = deltas[1:]
    x = np.arange(len(deltas.index)); bottom = np.zeros(len(deltas), dtype=float)
    plt.figure(figsize=(12, 6)); ax = plt.gca()
    for p in deltas.columns:
        vals = deltas[p].fillna(0.0).values
        ax.bar(x, vals, bottom=bottom, label=str(p))
        bottom = bottom + vals
    ax.set_xticks(x)
    try: ax.set_xticklabels([int(v) for v in deltas.index], rotation=0)
    except Exception: ax.set_xticklabels(list(deltas.index), rotation=0)
    ax.set_xlabel("Iteration"); ax.set_ylabel("Fractional Δ of width (1 = +100%)")
    ax.set_title("Stacked fractional change of search-space dimensions per iteration")
    ax.legend(loc="best", ncols=2); plt.tight_layout()
    plt.savefig(out_png, dpi=200, bbox_inches="tight"); plt.close()
    return out_png

# --- "Wrapper" function called by main ---

def run_per_experiment_analysis(experiment_dir: Path, run_train: bool = False):
    """
    Runs all per-experiment analyses (log copying, plotting, training).
    This is the function called by 'analyze.py' if --skip-plots is not used.
    """
    rep_log = utils.pick_representative_log(experiment_dir)
    output_log = None
    if rep_log:
        output_log = utils.copy_log_to_output(rep_log, experiment_dir)
    
    if output_log:
        try:
            csv_evo = write_search_space_evolution(experiment_dir, output_log)
            if csv_evo and csv_evo.exists():
                build_stacked_fractional_change_plot(
                    csv_path=csv_evo,
                    out_png=experiment_dir / "search_space_pct_changes.png",
                    out_csv=experiment_dir / "search_space_pct_changes.csv"
                )
        except Exception as e:
            logging.warning("Could not generate search space analysis for %s: %s", experiment_dir.name, e)

    if run_train:
        acc_test, loss_test = train_best_model_if_required(experiment_dir)
        if acc_test:
            logging.info("Test Accuracy (fine-tuned) for %s: %.4f", experiment_dir.name, acc_test)