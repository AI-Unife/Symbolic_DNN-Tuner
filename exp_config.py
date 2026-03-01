# exp_config.py
from __future__ import annotations
from dataclasses import dataclass, field
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

# ---------------- Schema con default (identici al parser di symbolic_tuner.py) ----------------
@dataclass
class ConfigSchema:
    backend: str = "tf"                          # tf | torch
    eval: int = 300                              # Max number of evaluations
    early_stop: int = 30                         # Early stopping patience
    epochs: int = 2                              # Epochs for training
    mod_list: List[str] = field(default_factory=list)  # List of active modules
    dataset: str = "light"                       # Dataset name
    name: str = "experiment"                     # Experiment name
    seed: int = 42                               # Random seed
    quantization: bool = False                   # Quantize the network
    verbose: int = 2                             # Verbosity level (0: silent, 1: space, 2: space+model)
    w_FLOPS: float = 0.33                        # Weight Flops loss
    w_HW: float = 0.33                           # Weight HW loss
    lacc: float = 0.10                           # Accuracy loss threshold (underfitting)
    flops_th: int = 150000000                    # Max number of FLOPS
    nparams_th: int = 2500000                    # Max number of PARAMS
    opt: str = "filtered"                        # Optimizer type (standard | filtered | basic | RS | RS_ruled)
    use_hw_cost: bool = True                     # Flag to use or not hw cost in simulation
    hw_backend: str = "nvdla"                    # hw backend for profiling, "nvdla" or "ember"
    suggest_net_opt: bool = False                # flag to accept suggestions for network optimization
    suggest_hw_opt: bool = False                 # flag to accept suggestions for hardware optimization

# ---------------- Validazione (stesse regole del parser) ----------------
_VALID_MODULES = {"hardware_module", "flops_module"}
_VALID_OPT = {"standard", "filtered", "basic", "RS", "RS_ruled"}
_VALID_BACKENDS = {"tf", "torch"}
_VALID_HW_BACKENDS = {"nvdla", "ember"}

def create_config_file(exp_dir: str | Path, overrides: Optional[Dict[str, Any]] = None) -> Path:
    """
    Crea (o sovrascrive) un file config.yaml nella cartella dell'esperimento.
    Usa i default dello schema e applica gli override passati come dict.
    """
    p = Path(exp_dir).expanduser().resolve()
    p.mkdir(parents=True, exist_ok=True)
    cfg_path = p / "config.yaml"

    # defaults + override
    _validate(overrides or {})
    schema = ConfigSchema()
    base = {
        "backend": schema.backend,
        "eval": schema.eval,
        "early_stop": schema.early_stop,
        "epochs": schema.epochs,
        "mod_list": schema.mod_list,
        "dataset": schema.dataset,
        "name": schema.name,
        "seed": schema.seed,
        "quantization": schema.quantization,
        "verbose": schema.verbose,
        "w_FLOPS": schema.w_FLOPS,
        "w_HW": schema.w_HW,
        "lacc": schema.lacc,
        "flops_th": schema.flops_th,
        "nparams_th": schema.nparams_th,
        "opt": schema.opt,
        "use_hw_cost": schema.use_hw_cost,
        "hw_backend": schema.hw_backend,
        "suggest_net_opt": schema.suggest_net_opt,
        "suggest_hw_opt": schema.suggest_hw_opt
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
    if isinstance(mods, str):  # In case it's a string, convert to list
        mods = [mods]
    bad = [m for m in mods if m not in _VALID_MODULES]
    if bad:
        raise ValueError(f"Invalid module(s) {bad}. Choose from: {sorted(_VALID_MODULES)}")
    
    # opt
    opt = d.get("opt", "filtered")
    if opt not in _VALID_OPT:
        print(f"WARNING: Invalid opt '{opt}'. Choose from: {sorted(_VALID_OPT)}. Set to 'filtered'")
        d['opt'] = 'filtered'
    
    # backend
    backend = d.get("backend", "tf")
    if backend not in _VALID_BACKENDS:
        print(f"WARNING: Invalid backend '{backend}'. Choose from: {sorted(_VALID_BACKENDS)}. Set to 'tf'")
        d['backend'] = 'tf'

    # hw_backend
    hw = d.get("hw_backend", "nvdla")
    if hw not in _VALID_HW_BACKENDS:
        print(f"WARNING: Invalid hw_backend '{hw}'. Choose from: {sorted(_VALID_HW_BACKENDS)}. Set to 'nvdla'")
        d["hw_backend"] = "nvdla"

    

# ---------------- Loader + discovery ----------------
_ENV_KEY = "EXP_CONFIG"   # puoi impostarlo per puntare al config della run
_lock = threading.Lock()
_cached_path: Optional[Path] = None
_cached_mtime: Optional[float] = None
_cached_cfg: Optional[DotDict] = None

def _discover_config_path() -> Path:
    """
    Trova il config.yaml attivo. Se non esiste, lo crea automaticamente
    nella directory corrente o in quella indicata da EXP_CONFIG.
    """
    # 1) ENV EXP_CONFIG
    env = os.getenv(_ENV_KEY)
    if env:
        p = Path(env).expanduser().resolve()
        if p.is_file():
            return p
        elif p.parent.exists():
            print(f"[exp_config] config.yaml non trovato, lo creo in: {p}")
            return create_config_file(p.parent)
        else:
            raise FileNotFoundError(f"Directory non valida per EXP_CONFIG: {p.parent}")

    # 2) Fallback: directory corrente
    p = Path.cwd() / "config.yaml"
    if p.is_file():
        return p
    else:
        print(f"[exp_config] config.yaml non trovato, lo creo in: {Path.cwd()}")
        return create_config_file(Path.cwd())
    

def _read_yaml(path: Path) -> Dict[str, Any]:
    data = yaml.safe_load(path.read_text()) or {}
    if not isinstance(data, dict):
        raise ValueError(f"{path} deve essere un mapping YAML (dict).")
    return data

def _apply_defaults(d: Dict[str, Any]) -> Dict[str, Any]:
    """Riempi con i default dello schema."""
    schema = ConfigSchema()
    merged = {
        "backend": d.get("backend", schema.backend),
        "eval": d.get("eval", schema.eval),
        "early_stop": d.get("early_stop", schema.early_stop),
        "epochs": d.get("epochs", schema.epochs),
        "mod_list": d.get("mod_list", schema.mod_list),
        "dataset": d.get("dataset", schema.dataset),
        "name": d.get("name", schema.name),
        "seed": d.get("seed", schema.seed),
        "quantization": d.get("quantization", schema.quantization),
        "verbose": d.get("verbose", schema.verbose),
        "w_FLOPS": d.get("w_FLOPS", schema.w_FLOPS),
        "w_HW": d.get("w_HW", schema.w_HW),
        "lacc": d.get("lacc", schema.lacc),
        "flops_th": d.get("flops_th", schema.flops_th),
        "nparams_th": d.get("nparams_th", schema.nparams_th),
        "opt": d.get("opt", schema.opt),
        "use_hw_cost": d.get("use_hw_cost", schema.use_hw_cost),
        "hw_backend": d.get("hw_backend", schema.hw_backend),
        "suggest_net_opt": d.get("suggest_net_opt", schema.suggest_net_opt),
        "suggest_hw_opt": d.get("suggest_hw_opt", schema.suggest_hw_opt)
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