# exp_config.py
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional, Any, Dict
from pathlib import Path
import os, threading, yaml
import datetime
# ---------------- Dot-dict semplice ----------------
class DotDict(dict):
    __getattr__ = dict.get
    def __setattr__(self, k, v): self[k] = v
    def freeze(self):
        for k, v in list(self.items()):
            if isinstance(v, dict) and not isinstance(v, DotDict):
                self[k] = DotDict(v).freeze()
        return self

# ---------------- Schema con default (identici al tuo argparse) ----------------
@dataclass
class ConfigSchema:
    eval: int = 30
    epochs: int = 2
    mod_list: List[str] = None
    dataset: str = "cifar-10"
    name: str = "debug"
    frames: int = 16
    mode: str = "fwdPass"            # fwdPass | depth | hybrid
    channels: int = 2
    polarity: str = "both"           # both | sum | sub | drop
    seed: int = 42
    opt: str = "RS_ruled"            # standard | filtered | basic | RS | RS_ruled

# ---------------- Validazione (stesse regole del tuo parser) ----------------
_VALID_MODULES = {"hardware_module", "flops_module"}
_VALID_MODES = {"fwdPass", "depth", "hybrid"}
_VALID_POLARITY = {"both", "sum", "sub", "drop"}
_VALID_OPT = {"standard", "filtered", "basic", "RS", "RS_ruled"}

def create_config_file(exp_dir: str | Path, overrides: Optional[Dict[str, Any]] = None) -> Path:
    """
    Crea (o sovrascrive) un file config.yaml nella cartella dell'esperimento.
    Usa i default dello schema e applica gli override passati come dict.
    """
    p = Path(exp_dir).expanduser().resolve()
    p.mkdir(parents=True, exist_ok=True)
    cfg_path = p / "config.yaml"

    # defaults + override
    _validate(overrides)
    schema = ConfigSchema()
    base = {
        "eval": schema.eval,
        "epochs": schema.epochs,
        "mod_list": schema.mod_list or [],
        "dataset": schema.dataset,
        "name": schema.name,
        "frames": schema.frames,
        "mode": schema.mode,
        "channels": schema.channels,
        "polarity": schema.polarity,
        "seed": schema.seed,
        "opt": schema.opt,
        "verbose": schema.verbose if hasattr(schema, "verbose") else 0,
        "quantization": schema.quantization if hasattr(schema, "quantization") else False,
        "early_stop": schema.early_stop if hasattr(schema, "early_stop") else 20,
        "dataset_path": schema.dataset_path if hasattr(schema, "dataset_path") else "./data",
        "cache_dataset": schema.cache_dataset if hasattr(schema, "chache_dataset") else "./cache",
    }
    if overrides:
        base.update(overrides)


    # timestamp utile per tracciabilità
    base["created_at"] = datetime.datetime.now().isoformat(timespec="seconds")

    # salva YAML
    import yaml
    with open(cfg_path, "w") as f:
        yaml.safe_dump(base, f, sort_keys=False, allow_unicode=True)

    print(f"[exp_config] Creato config.yaml in: {cfg_path}")
    return cfg_path

def _validate(d: Dict[str, Any]) -> None:
    # mod_list
    mods = d.get("mod_list") or []
    bad = [m for m in mods if m not in _VALID_MODULES]
    if bad:
        raise ValueError(f"Invalid module(s) {bad}. Choose from: {sorted(_VALID_MODULES)}")
    # mode
    if d.get("mode") not in _VALID_MODES:
        raise ValueError(f"Invalid mode '{d.get('mode')}'. Choose from: {sorted(_VALID_MODES)}")
    # polarity
    if d.get("polarity") not in _VALID_POLARITY:
        raise ValueError(f"Invalid polarity '{d.get('polarity')}'. Choose from: {sorted(_VALID_POLARITY)}")
    # opt
    if d.get("opt") not in _VALID_OPT:
        print(f"Invalid opt '{d.get('opt')}'. Choose from: {sorted(_VALID_OPT)}. Set RS_ruled")
        d['opt'] = 'RS_ruled'

# ---------------- Loader + discovery ----------------
_ENV_KEY = "EXP_CONFIG"   # puoi impostarlo per puntare al config della run
_lock = threading.Lock()
_cached_path: Optional[Path] = None
_cached_mtime: Optional[float] = None
_cached_cfg: Optional[DotDict] = None

def _discover_config_path() -> Path:
    # 1) ENV EXP_CONFIG
    env = os.getenv(_ENV_KEY)
    if env:
        p = Path(env).expanduser().resolve()
        if p.is_file():
            return p
    raise FileNotFoundError("config.yaml non trovato. Imposta ENV EXP_CONFIG o lancia da dentro la cartella esperimento.")

def _read_yaml(path: Path) -> Dict[str, Any]:
    data = yaml.safe_load(path.read_text()) or {}
    if not isinstance(data, dict):
        raise ValueError(f"{path} deve essere un mapping YAML (dict).")
    return data

def _apply_defaults(d: Dict[str, Any]) -> Dict[str, Any]:
    # riempi con i default dello schema
    schema = ConfigSchema()
    merged = {
        "eval": d.get("eval", schema.eval),
        "epochs": d.get("epochs", schema.epochs),
        "mod_list": d.get("mod_list", schema.mod_list or []),
        "dataset": d.get("dataset", schema.dataset),
        "name": d.get("name", schema.name),
        "frames": d.get("frames", schema.frames),
        "mode": d.get("mode", schema.mode),
        "channels": d.get("channels", schema.channels),
        "polarity": d.get("polarity", schema.polarity),
        "seed": d.get("seed", schema.seed),
        "opt": d.get("opt", schema.opt),
        "verbose": d.get("verbose", getattr(schema, "verbose", 0)),
        "quantization": d.get("quantization", getattr(schema, "quantization", False)),
        "early_stop": d.get("early_stop", getattr(schema, "early_stop", 20)),
        "dataset_path": d.get("dataset_path", getattr(schema, "dataset_path", "./data")),
        "cache_dataset": d.get("cache_dataset", getattr(schema, "cache_dataset", "./cache")),
    }
    return merged

def load_cfg_from_path(path: str | Path, *, validate: bool = True) -> DotDict:
    p = Path(path).expanduser().resolve()
    d = _read_yaml(p)
    d = _apply_defaults(d)
    if validate:
        _validate(d)
    return DotDict(d).freeze()

def load_cfg(force: bool = False) -> DotDict:
    """Carica (e cache-a) il config 'attivo' secondo le regole di discovery."""
    global _cached_path, _cached_mtime, _cached_cfg
    with _lock:
        path = _discover_config_path()
        mtime = path.stat().st_mtime
        if force or _cached_cfg is None or _cached_path != path or (_cached_mtime and mtime > _cached_mtime):
            d = _read_yaml(path)
            d = _apply_defaults(d)
            _validate(d)
            _cached_cfg = DotDict(d).freeze()
            _cached_path = path
            _cached_mtime = mtime
        return _cached_cfg

def set_active_config(path: str | Path) -> None:
    """Imposta (per il processo corrente) il file di config da usare."""
    p = Path(path).expanduser().resolve()
    os.environ[_ENV_KEY] = str(p)


def reload_cfg() -> DotDict:
    return load_cfg(force=True)