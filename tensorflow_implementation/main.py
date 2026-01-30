import os
import sys
from pathlib import Path 
import argparse

# Add project root to sys.path so package imports work when running as a script
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from tensorflow.keras import backend as K
from components.dataset import TunerDataset
from tensorflow_implementation import module_backend, neural_network
from components.tuner import Tuner, TunerConfig

from exp_config import create_config_file, set_active_config, load_cfg


os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")


def build_config() -> TunerConfig:
    dataset = TunerDataset()
    dataset.load_cifar_10()

    return TunerConfig(
        neural_network_cls=neural_network.NeuralNetwork,
        module_backend_cls=module_backend.ModuleBackend,
        dataset=dataset,
        clear_session_callback=K.clear_session
    )


def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments for the Symbolic DNN Tuner configuration.

    Returns:
        argparse.Namespace containing all configuration values.
    """
    parser = argparse.ArgumentParser(
        description="Symbolic DNN Tuner Configuration"
    )

    parser.add_argument("--eval", type=int, default=300,
                        help="Max number of evaluations")
    parser.add_argument("--early_stop", type=int, default=30,
                        help="Early stopping patience")
    parser.add_argument("--epochs", type=int, default=2,
                        help="Epochs for training")
    parser.add_argument(
        "--mod_list", nargs="+", default=[],
        help="List of active modules (e.g., hardware_module flops_module)"
    )
    parser.add_argument("--dataset", type=str, default="cifar10",
                        help="Dataset name")
    parser.add_argument("--name", type=str, default="debug",
                        help="Experiment name")
    parser.add_argument("--frames", type=int, default=16,
                        help="Number of frames for gesture dataset")
    parser.add_argument("--mode", type=str, default="fwdPass",
                        choices=["fwdPass", "depth", "hybrid"],
                        help="Experiment mode (fwdPass, depth, hybrid)")
    parser.add_argument("--channels", type=int, default=2,
                        help="Number of channels for the dataset")
    parser.add_argument("--polarity", type=str, default="both",
                        choices=["both", "sum", "sub", "drop"],
                        help="Polarity for event-based datasets")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")
    parser.add_argument('--quantization', action='store_true',
                        help='quantize the network')

    parser.add_argument("--verbose", type=int, default=2, 
                        help="Verbosity level (0: silent, 1: print space, 2: print space and model summary)")
    parser.add_argument(
        "--opt", type=str, default="filtered",
        choices=["standard", "filtered", "basic", "RS", "RS_ruled"],
        help="Optimizer type for the analysis"
    )
    
    parser.add_argument("--dataset_path", type=str, default="/hpc/home/bzzlca/AIDA4Edge/data/",
                        help="Gesture Dataset Dir name")
    
    parser.add_argument("--cache_dataset", type=str, default="/hpc/home/bzzlca/AIDA4Edge/tf/",
                        help="Gesture Dataset Cache Dir name")

    args = parser.parse_args()

    # Validate module list
    valid_modules = {"hardware_module", "flops_module"}
    for mod in args.mod_list:
        if mod not in valid_modules:
            parser.error(
                f"Invalid module '{mod}'. Choose from: {', '.join(valid_modules)}"
            )

    return args

def main():
    # --- 1. Configuration Setup ---
    args = parse_args()
    
    exp_dir = Path(args.name)
    exp_dir.mkdir(parents=True, exist_ok=True)
    
    cfg_path = create_config_file(exp_dir, overrides=args.__dict__)
    set_active_config(cfg_path)
    cfg = load_cfg(force=True)
        
    tuner = Tuner(build_config())
    tuner.run()


if __name__ == "__main__":
    main()
