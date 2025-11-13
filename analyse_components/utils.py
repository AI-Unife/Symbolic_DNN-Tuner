# In utils.py
import shutil
import logging
from pathlib import Path
from typing import Optional

def extract_tuner(exp_name: str) -> Optional[str]:
    """Heuristically infers the tuner name from the experiment folder string."""
    tuner_names = ["filtered", "basic", "standard", "RS_ruled", "RS"]
    for t in tuner_names:
        if t in exp_name: return t
    parts = exp_name.split("_")
    if len(parts) > 7:
        cand = parts[6]
        if parts[7] == "ruled": cand += "_ruled"
        return cand
    return None

def copy_log_to_output(log_src: Path, exp_dir: Path) -> Optional[Path]:
    """Copies a representative log to <exp_dir>/output/output.log for inspection."""
    dest_dir = exp_dir / "output"; dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / "output.log"
    try:
        shutil.copy2(log_src, dest)
        logging.info("Copied log: %s -> %s", log_src, dest)
        return dest
    except Exception as e:
        logging.warning("Failed to copy %s to %s: %s", log_src, dest, e)
        return None

def pick_representative_log(exp_dir: Path) -> Optional[Path]:
    """Chooses a log to copy: prefers 'output.log', falls back to largest .out file."""
    explicit = exp_dir / "output.log"
    if explicit.is_file(): return explicit
    outs = list(exp_dir.rglob("*.out"))
    if not outs: return None
    # Pick the largest by size
    outs.sort(key=lambda p: p.stat().st_size if p.exists() else 0, reverse=True)
    return outs[0]

def should_select_experiment_dir(dir_name: str, prefix: str) -> bool:
    """Return True if the folder should be considered an experiment."""
    return dir_name.startswith(prefix)