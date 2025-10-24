#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
extract_results.py

- Scansiona tutte le cartelle esperimento che iniziano con '25_'.
- Estrae SOLO dai log in algorithm_logs/:
    * acc_report.txt -> Acc_Best, Acc_Last, Acc_Mean
    * <module>_report.txt (es. flops_report.txt, hardware_report.txt, ...) :
        ogni riga può contenere 1+ numeri separati da virgola;
        per OGNI colonna calcola: <module>_col{i}_{last,mean,best}
- Per lo "search space" usa il file .out (dentro la cartella esperimento),
  costruendo search_space_evolution.(csv/xlsx) e search_space_pct_changes.(csv/xlsx).
- Salvataggi con pandas: CSV + Excel (.xlsx).

Output globali:
  - total_results.(csv/xlsx)
  - mean_results.(csv/xlsx)
Output per-esperimento:
  - search_space_evolution.(csv/xlsx)
  - search_space_pct_changes.(csv/xlsx)
"""
from __future__ import annotations

# --- anti-shadowing di 'config' ---
import sys, types
if "config" not in sys.modules:
    sys.modules["config"] = types.ModuleType("config")
    # opzionale: evita che il tuo vero myconfig venga pescato più tardi
    sys.modules["config"].__file__ = "<stubbed>"
    

import argparse
import logging
import math
import re
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
import tensorflow as tf
import ast
# ------------------------------------------------------------------------------
# Logging
# ------------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
)

# ------------------------------------------------------------------------------
# Helpers comuni
# ------------------------------------------------------------------------------
ANSI_RE = re.compile(r"\x1B\[[0-?]*[ -/]*[@-~]")

def _strip_ansi(s: str) -> str:
    return ANSI_RE.sub("", s)

def _find_out_file(exp_dir: Path) -> Optional[Path]:
    """Restituisce il primo file .out trovato nella cartella esperimento."""
    cands = sorted(exp_dir.glob("*.out"))
    return cands[0] if cands else None

def _parse_csv_numbers_line(line: str) -> List[float]:
    parts = [p.strip() for p in line.strip().split(",")]
    vals: List[float] = []
    for p in parts:
        if not p:
            continue
        try:
            vals.append(float(p))
        except ValueError:
            # ignora token non numerici
            pass
    return vals

def _extract_tuner(exp_name: str) -> Optional[str]:
    # euristica semplice; adatta ai tuoi nomi
    known = ["filtered", "basic", "standard", "RS_ruled", "RS"]
    for k in known:
        if k in exp_name:
            return k
    # fallback euristico
    parts = exp_name.split("_")
    return parts[6] if len(parts) > 6 else None

def _extract_dataset(exp_name: str) -> Optional[str]:
    # euristica semplice; adatta ai tuoi nomi
    known = ["cifar10", "cifar100", "gesture", "roigesture", "imagenet16120"]
    exp_name = exp_name.lower().replace("-", "").split("_")
    for k in known:
        for part in exp_name:
            if k == part:
                return k
    # fallback euristico
    parts = exp_name.split("_")
    return parts[5] if len(parts) > 5 else None

def get_best_iteration_from_acc(aldir: Path) -> Optional[int]:
    """Restituisce l'indice (1-based) della riga con accuracy massima in acc_report.txt."""
    acc_path = aldir / "acc_report.txt"
    if not acc_path.exists():
        return None
    vals = []
    for ln in acc_path.read_text().splitlines():
        ln = ln.strip()
        if not ln:
            continue
        try:
            vals.append(float(ln))
        except ValueError:
            pass
    if not vals:
        return None
    best_idx = int(pd.Series(vals).idxmax())  # 0-based posizione
    # Converte in 1-based per coerenza con "iteration" dei report
    return best_idx + 1

# ------------------------------------------------------------------------------
# Retrain Network best-model.keras 
# ------------------------------------------------------------------------------

def parse_batch_opt_lr_from_hyper_neural(aldir: Path, iteration: int) -> Optional[str]:
    """
    Fallback: prendi la riga 'iteration' da hyper-neural.txt (1-based) e leggi il campo 'optimizer'.
    """
    f = aldir / "hyper-neural.txt"
    if not f.exists():
        return None
    rows = []
    for ln in f.read_text().splitlines():
        ln = ln.strip()
        if not ln:
            continue
        try:
            d = ast.literal_eval(ln)
            if isinstance(d, dict):
                rows.append(d)
        except Exception:
            continue
    if not rows:
        return None
    idx = max(0, min(iteration - 1, len(rows) - 1))
    batch = rows[idx].get("batch_size")
    opt = rows[idx].get("optimizer")
    lr = rows[idx].get("learning_rate")
    return (batch, opt, lr)

def load_dataset(dataset_name: str):
    """
    load_dataset(dataset_name) -> (train_ds, val_ds).
    """
    # ds = get_datasets(dataset_name)

    # # Heuristics: molti loader restituiscono tuple (train, val) o 4-tuple
    # if isinstance(ds, tuple):
    #     if len(ds) == 2:
    #         train_ds, val_ds = ds
    #         return train_ds, val_ds
    #     elif len(ds) == 4:
    #         x_tr, y_tr, x_va, y_va = ds
    #         return (x_tr, y_tr), (x_va, y_va)
    # # altrimenti assumiamo sia già un tf.data.Dataset con split nel dict
    # if isinstance(ds, dict) and "train" in ds and "val" in ds:
    #     return ds["train"], ds["val"]
    # raise RuntimeError("Formato dataset non riconosciuto")
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
    # Convert class vectors to binary class matrices.
    y_train = tf.keras.utils.to_categorical(y_train, 10)
    y_test = tf.keras.utils.to_categorical(y_test, 10)
    return (x_train, y_train), (x_test, y_test)

def make_optimizer_by_name(name: str, lr: float = 1e-3):
    name_low = (name or "").lower()
    if name_low in ("adamw",):
        try:
            return tf.keras.optimizers.AdamW(learning_rate=lr, weight_decay=0.0)
        except Exception:
            # fallback su Adam
            return tf.keras.optimizers.Adam(learning_rate=lr)
    if name_low in ("adamax",):
        return tf.keras.optimizers.Adamax(learning_rate=lr)
    if name_low in ("adam",):
        return tf.keras.optimizers.Adam(learning_rate=lr)
    if name_low in ("sgd",):
        return tf.keras.optimizers.SGD(learning_rate=lr, momentum=0.9, nesterov=True)
    if name_low in ("rmsprop",):
        return tf.keras.optimizers.RMSprop(learning_rate=lr, momentum=0.9)
    # default
    return tf.keras.optimizers.Adam(learning_rate=lr)

    
    
def train_best_model_if_required(exp_dir: Path, dataset_name: str, aldir: Path):
    model_path = exp_dir / "Model" / "best-model.keras"
    if not model_path.exists():
        logging.warning("best-model.keras non trovato in %s", model_path)
        return
    model = tf.keras.models.load_model(model_path)
    # 1) Iterazione migliore
    best_it = get_best_iteration_from_acc(aldir)
    if not best_it:
        logging.warning("Impossibile determinare l'iterazione migliore da acc_report.txt")
        return None, None

    # 2) Optimizer lr and batch size dalla riga best_it di hyper-neural.txt
    opt_name = None
    lr = None
    batch = None
    (batch, opt_name, lr) = parse_batch_opt_lr_from_hyper_neural(aldir, best_it) or (None, None, None)
    if not opt_name:
        opt_name = "Adam"  # fallback sicuro
    if lr is None:
        lr = 1e-3  # default
    if batch is None:
        batch = 32  # default
    # 3) Load Dataset 
    train_split, val_split = load_dataset(dataset_name)

    # 4) Carica e compila
    optimizer = make_optimizer_by_name(opt_name, lr=lr)
    loss = tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.0)
    metrics = [tf.keras.metrics.CategoricalAccuracy(name="acc")]

    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

    # 5) Allenamento breve di “refit” (configurabile se vuoi aggiungere argomenti CLI)
    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor="val_acc", mode="max", patience=100, restore_best_weights=True)
    ]

    history = None
    if isinstance(train_split, tuple):
        (x_tr, y_tr) = train_split
        (x_va, y_va) = val_split
        history = model.fit(x_tr, y_tr, validation_data=(x_va, y_va), epochs=1000, batch_size=batch, callbacks=callbacks, verbose=2)
    else:
        # tf.data
        history = model.fit(train_split, validation_data=val_split, epochs=1000, callbacks=callbacks, verbose=2)

    # 6) Salva modello e storia
    finetuned_path = exp_dir / "Model" / "best-model-finetuned.keras"
    try:
        model.save(finetuned_path)
    except Exception as e:
        logging.warning("Salvataggio modello finetuned fallito: %s", e)

    # storia su CSV
    if history is not None:
        hist_df = pd.DataFrame(history.history)
        hist_csv = exp_dir / "best_model_finetune_history.csv"
        hist_df.to_csv(hist_csv, index=False)
        
    logging.info("Training best model completato. Optimizer=%s, LR=%.3g", opt_name, lr)
    # 7) Valutazione su test set (se disponibile)
    acc_test, loss_test = model.evaluate(val_split, verbose=2) if not isinstance(val_split, tuple) else model.evaluate(x_va, y_va, verbose=2)
    return acc_test, loss_test
# ------------------------------------------------------------------------------
# Search space: parser .out  -> evolution + pct changes
# ------------------------------------------------------------------------------
def _iter_search_space_blocks(out_path: Path):
    """
    Genera blocchi 'Search space exploration report (iteration X)'
    Restituisce (iterazione:int, [righe "Dimension ..."])
    """
    try:
        lines = out_path.read_text(errors="ignore").splitlines()
    except FileNotFoundError:
        return
    i, n = 0, len(lines)
    while i < n:
        line = _strip_ansi(lines[i]).strip()
        m = re.match(r"Search space exploration report\s*\(iteration\s*(\d+)\)", line, re.I)
        if m:
            iter_no = int(m.group(1))
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
    Parse: 'Dimension k: name - TYPE(low=..., high=..., ...)'
    Ritorna dict con iteration (dummy, sovrascritta), param, kind, low, high, width.
    """
    m = re.match(r"Dimension\s+(\d+):\s*([^-\s]+)\s*-\s*([A-Za-z]+)\((.*)\)$", line)
    if not m:
        return None
    idx = int(m.group(1))
    name = m.group(2)
    kind = m.group(3)
    inner = m.group(4)
    low = high = None
    mlow = re.search(r"low\s*=\s*([-\d\.eE]+)", inner)
    mhigh = re.search(r"high\s*=\s*([-\d\.eE]+)", inner)
    if mlow:
        try:
            low = float(mlow.group(1))
        except Exception:
            pass
    if mhigh:
        try:
            high = float(mhigh.group(1))
        except Exception:
            pass
    width = None
    if low is not None and high is not None:
        width = abs(high - low)
    return {"iteration": idx, "param": name, "kind": kind, "low": low, "high": high, "width": width}

def write_search_space_evolution_from_out(exp_dir: Path, out_path: Path) -> Optional[Path]:
    rows = []
    for it, block in _iter_search_space_blocks(out_path):
        for ln in block:
            d = _parse_dimension_line(ln)
            if d is not None:
                d["iteration"] = it
                rows.append(d)
    if not rows:
        return None
    df = pd.DataFrame(rows).sort_values(["iteration", "param"])
    csv_path = exp_dir / "search_space_evolution.csv"
    xlsx_path = exp_dir / "search_space_evolution.xlsx"
    df.to_csv(csv_path, index=False)
    try:
        df.to_excel(xlsx_path, index=False)
    except Exception as e:
        logging.warning("Excel write failed for %s: %s", xlsx_path, e)
    return csv_path

def build_fractional_changes(csv_path: Path, out_csv: Path, out_xlsx: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    # risolvi i nomi ignorando maiuscole/minuscole
    def _resolve(col: str) -> str:
        low = col.lower()
        for c in df.columns:
            if c.lower() == low:
                return c
        raise KeyError(col)
    col_iter = _resolve("iteration")
    col_param = _resolve("param")
    col_width = _resolve("width")

    piv = (
        df[[col_iter, col_param, col_width]]
        .pivot_table(index=col_iter, columns=col_param, values=col_width, aggfunc="last")
        .sort_index()
        .fillna(0.0)
    )

    deltas = []
    prev = None
    for idx, row in piv.iterrows():
        if prev is None:
            deltas.append(pd.Series(1.0, index=piv.columns, name=idx))
        else:
            r = []
            for c in piv.columns:
                a, b = float(prev[c]), float(row[c])
                if a == 0.0 and b > 0.0:
                    r.append(1.0)
                elif a == 0.0 and b == 0.0:
                    r.append(0.0)
                else:
                    r.append((b - a) / a)
            deltas.append(pd.Series(r, index=piv.columns, name=idx))
        prev = row

    delta_df = pd.DataFrame(deltas)
    delta_df.to_csv(out_csv, index=True)
    try:
        delta_df.to_excel(out_xlsx, index=True)
    except Exception as e:
        logging.warning("Excel write failed for %s: %s", out_xlsx, e)
    return delta_df

# ------------------------------------------------------------------------------
# Parser algorithm_logs
# ------------------------------------------------------------------------------
@dataclass
class ExperimentRow:
    Run: str
    Path: str
    Tuner: str = ""
    Dataset: str = ""  # valorizzala se la ricavi dal nome / struttura
    Acc_Best: float = math.nan
    Acc_Last: float = math.nan
    Acc_Mean: float = math.nan
    Extra: Dict[str, float] = field(default_factory=dict)

    def to_series(self) -> pd.Series:
        base = asdict(self)
        extra = base.pop("Extra", {})
        base.update(extra)
        return pd.Series(base)

def parse_acc_report(path: Path) -> Tuple[float, float, float]:
    if not path.exists():
        return (math.nan, math.nan, math.nan)
    vals: List[float] = []
    for ln in path.read_text().splitlines():
        ln = ln.strip()
        if not ln:
            continue
        try:
            vals.append(float(ln))
        except ValueError:
            pass
    if not vals:
        return (math.nan, math.nan, math.nan)
    return max(vals)

def parse_module_report(module_name: str, path: Path) -> Dict[str, float]:
    """
    <module>_report.txt: N numeri separati da ',' per riga.
    Costruisce colonne: <module>_col{i}_{last,mean,best} per i=1..N
    """
    out: Dict[str, float] = {}
    if not path.exists():
        return out
    rows: List[List[float]] = []
    for ln in path.read_text().splitlines():
        if not ln.strip():
            continue
        vals = _parse_csv_numbers_line(ln)
        if vals:
            rows.append(vals)
    if not rows:
        return out
    width = max(len(r) for r in rows)
    # pad con NaN per righe corte
    for r in rows:
        if len(r) < width:
            r += [math.nan] * (width - len(r))
    cols = list(zip(*rows))  # col 0..width-1
    for i, col in enumerate(cols, start=1):
        s = pd.to_numeric(pd.Series(col, dtype=float), errors="coerce")
        s_no_nan = s.dropna()
        last = float(s_no_nan.iloc[-1]) if not s_no_nan.empty else math.nan
        out[f"{module_name}_col{i}_last"] = last
        out[f"{module_name}_col{i}_mean"] = float(s.mean())
        out[f"{module_name}_col{i}_best"] = float(s.max())
    return out

def summarize_experiment(exp_dir: Path) -> ExperimentRow:
    row = ExperimentRow(Run=exp_dir.name, Path=str(exp_dir))
    row.Tuner = _extract_tuner(exp_dir.name) or ""
    row.Dataset = _extract_dataset(exp_dir.name) or ""

    # --- algorithm_logs
    aldir = exp_dir / "algorithm_logs"
    if not aldir.exists():
        logging.warning("Missing algorithm_logs in %s", exp_dir)
        return row

    # acc_report
    best = parse_acc_report(aldir / "acc_report.txt")
    row.Acc_Best = best

    # altri moduli: *_report.txt (tranne acc_report.txt)
    for p in sorted(aldir.glob("*_report.txt")):
        if p.name == "acc_report.txt":
            continue
        if not p.stem.endswith("_report"):
            continue
        module = p.stem[:-len("_report")]
        row.Extra.update(parse_module_report(module, p))

    # --- Search space dallo .out
    out_file = _find_out_file(exp_dir)
    if out_file:
        try:
            evo_csv = write_search_space_evolution_from_out(exp_dir, out_file)
            if evo_csv:
                build_fractional_changes(
                    csv_path=evo_csv,
                    out_csv=exp_dir / "search_space_pct_changes.csv",
                    out_xlsx=exp_dir / "search_space_pct_changes.xlsx",
                )
        except Exception as e:
            logging.warning("Search space parsing failed for %s: %s", exp_dir.name, e)

    return row

# ------------------------------------------------------------------------------
# Aggregazioni / salvataggi
# ------------------------------------------------------------------------------
def build_total_results(base_dir: Path, exp_prefix: str, out_csv: Path, out_xlsx: Path, train_best: bool) -> pd.DataFrame:
    rows: List[pd.Series] = []
    for child in sorted(base_dir.iterdir()):
        if not child.is_dir():
            continue
        if not child.name.startswith(exp_prefix):
            continue
        logging.info("Scanning %s", child.name)
        try:
            s = summarize_experiment(child).to_series()
            s = s.drop("Path")
            if train_best:
            # allena il best per ogni esperimento scoperto
                aldir = child / "algorithm_logs"
                dataset_name = _extract_dataset(child.name) or "cifar10"  # fallback
                acc_test, loss_test = train_best_model_if_required(child, dataset_name, aldir)
                s["Acc_Test_Finetuned"] = acc_test
                s["Loss_Test_Finetuned"] = loss_test
            rows.append(s)
        except Exception as e:
            logging.exception("Failed on %s: %s", child, e)

    if not rows:
        df = pd.DataFrame(columns=["Run", "Path", "Tuner", "Dataset", "Acc_Best", "Acc_Last", "Acc_Mean"])
    else:
        df = pd.DataFrame(rows).fillna(value=pd.NA)

    # Salva CSV + Excel
    df.to_csv(out_csv, index=False)
    try:
        df.to_excel(out_xlsx, index=False)
    except Exception as e:
        logging.warning("Excel write failed for %s: %s", out_xlsx, e)

    return df

def build_mean_results(total_df: pd.DataFrame, out_csv: Path, out_xlsx: Path) -> pd.DataFrame:
    df = total_df.copy()

    # (Opzionale) costruisci 'Modules' se in futuro tornano colonne *_module
    if "Modules" not in df.columns:
        df["Modules"] = "none"

    group_keys = ["Tuner", "Dataset", "Modules"]
    for k in group_keys:
        if k not in df.columns:
            df[k] = "unknown"

    # metriche numeriche
    num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    if not num_cols:
        # crea un placeholder per evitare errori
        num_cols = ["Acc_Best", "Acc_Last", "Acc_Mean"]
        for c in num_cols:
            if c not in df.columns:
                df[c] = pd.NA

    grouped = df.groupby(group_keys, dropna=False)
    agg_mean = grouped[num_cols].mean()
    agg_std  = grouped[num_cols].std(ddof=0)

    # combiniamo mean/std in una sola tabella con colonne "metric_mean" e "metric_std"
    out = agg_mean.copy()
    out.columns = [f"{c}_mean" for c in out.columns]
    std_ren = agg_std.copy()
    std_ren.columns = [f"{c}_std" for c in std_ren.columns]
    out = out.join(std_ren, how="outer")

    out = out.reset_index().sort_values(group_keys)

    # Salva CSV + Excel
    out.to_csv(out_csv, index=False)
    try:
        out.to_excel(out_xlsx, index=False)
    except Exception as e:
        logging.warning("Excel write failed for %s: %s", out_xlsx, e)

    return out

# ------------------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-dir", type=Path, default="./", help="Directory con esperimenti (25_*)")
    ap.add_argument("--exp-prefix", type=str, default="25_", help="Prefisso cartelle esperimento")
    ap.add_argument("--total-csv", type=Path, default=Path("total_results.csv"))
    ap.add_argument("--mean-csv", type=Path, default=Path("mean_results.csv"))
    ap.add_argument("--train-best", action="store_true", help="Se impostato, carica Model/best-model.keras e lo allena con optimizer/reg ottenuti dai log")
    args, _ = ap.parse_known_args()
    # impedisci ad altri parse_args() di vedere i tuoi flag
    import sys
    sys.argv = [sys.argv[0]]

    total_xlsx = args.total_csv.with_suffix(".xlsx")
    mean_xlsx  = args.mean_csv.with_suffix(".xlsx")


    # 1) total_results
    total_df = build_total_results(args.base_dir, args.exp_prefix, args.total_csv, total_xlsx, args.train_best)
    print(f"✅ Created: {args.total_csv} and {total_xlsx}")

    # 2) mean_results
    mean_df = build_mean_results(total_df, args.mean_csv, mean_xlsx)
    print(f"✅ Created: {args.mean_csv} and {mean_xlsx}")

if __name__ == "__main__":
    main()