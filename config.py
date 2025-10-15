import argparse
import sys

# ----------------------------- argument parsing -----------------------------

def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments for the Symbolic DNN Tuner configuration.

    Returns:
        argparse.Namespace containing all configuration values.
    """
    parser = argparse.ArgumentParser(
        description="Symbolic DNN Tuner Configuration"
    )

    parser.add_argument("--eval", type=int, default=30,
                        help="Max number of evaluations")
    parser.add_argument("--epochs", type=int, default=2,
                        help="Epochs for training")
    parser.add_argument(
        "--mod_list", nargs="+", default=[""],
        help="List of active modules (e.g., hardware_module flops_module)"
    )
    parser.add_argument("--dataset", type=str, default="cifar-10",
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
    parser.add_argument(
        "--opt", type=str, default="RS_ruled",
        choices=["standard", "filtered", "basic", "RS", "RS_ruled"],
        help="Optimizer type for the analysis"
    )

    args = parser.parse_args()

    # Validate module list
    valid_modules = {"hardware_module", "flops_module"}
    for mod in args.mod_list:
        if mod not in valid_modules:
            parser.error(
                f"Invalid module '{mod}'. Choose from: {', '.join(valid_modules)}"
            )

    return args


# ----------------------------- config extraction ----------------------------

args = parse_args()

MAX_EVAL = args.eval
EPOCHS = args.epochs
FRAMES = args.frames
MOD_LIST = args.mod_list
DATA_NAME = args.dataset
NAME_EXP = args.name
MODE = args.mode
NUM_CHANNELS = args.channels
POLARITY = args.polarity
SEED = args.seed
OPT = args.opt

# ----------------------------- optional summary -----------------------------

def print_config_summary() -> None:
    """Pretty-print the configuration summary."""
    print("\n========== Symbolic DNN Tuner Configuration ==========")
    print(f" Experiment Name  : {NAME_EXP}")
    print(f" Dataset          : {DATA_NAME}")
    print(f" Modules          : {', '.join(MOD_LIST)}")
    print(f" Max Evaluations  : {MAX_EVAL}")
    print(f" Epochs           : {EPOCHS}")
    print(f" Frames           : {FRAMES}")
    print(f" Mode             : {MODE}")
    print(f" Channels         : {NUM_CHANNELS}")
    print(f" Polarity         : {POLARITY}")
    print(f" Seed             : {SEED}")
    print(f" Optimizer        : {OPT}")
    print("=====================================================\n")

if __name__ == "__main__":
    print_config_summary()
