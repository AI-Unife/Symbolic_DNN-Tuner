"""
Constants for ModelBench - configuration options
"""
from components.dataset import Dataset2Class

# Dataset options (derived from components/dataset.py)
AVAILABLE_DATASETS = sorted(Dataset2Class.keys())

# Other options (mirror from exp_config.py for convenience)
AVAILABLE_MODES = {"fwdPass", "depth", "hybrid"}
AVAILABLE_POLARITY = {"both", "sum", "sub", "drop"}
AVAILABLE_OPTIMIZERS = {"standard", "filtered", "basic", "RS", "RS_ruled"}
AVAILABLE_MODULES = {"hardware_module", "flops_module"}