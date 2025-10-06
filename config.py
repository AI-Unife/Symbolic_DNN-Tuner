import argparse

# def get_experiment_name():
parser = argparse.ArgumentParser()
print("Symbolic DNN Tuner Configuration")
parser.add_argument("--eval", type=int, default=3, help="Max number of evaluations")
parser.add_argument("--epochs", type=int, default=2, help="Epochs for training")
parser.add_argument("--mod_list", nargs="+", default=["accuracy_module"],
                    help="Lista dei moduli separati da spazio (es: hardware_module accuracy_module)")
parser.add_argument("--dataset", type=str, default="ImageNet16-120", help="Dataset name")
parser.add_argument("--name", type=str, default="debug", help="Experiment name")
parser.add_argument("--frames", type=int, default=16, help="Number of frames for gesture dataset")
parser.add_argument("--mode", type=str, default="fwdPass", help="Mode for the experiment (fwdPass, depth or hybrid)")
parser.add_argument("--channels", type=int, default=2, help="Number of channels for the dataset")
parser.add_argument("--polarity", type=str, default="both", help="Polarity for the dataset (both, sum, sub, drop)")
parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
parser.add_argument("--opt",  type=str, help="which optimizer to use for the analysis", default="RS_ruled", choices=["standard", "filtered", "basic", "RS", "RS_ruled"])

args = parser.parse_args()

MAX_EVAL = args.eval
EPOCHS = args.epochs
FRAMES = args.frames
for mod in args.mod_list:
    if mod not in ["hardware_module", "accuracy_module", "flops_module"]:
        raise ValueError(f"Module {mod} invalid!, Please choose from hardware_module, accuracy_module, flops_module")

MOD_LIST = args.mod_list # ["hardware_module", "accuracy_module", "flops_module"]

DATA_NAME = args.dataset
NAME_EXP = args.name #f"{datetime.now().strftime('%y_%m_%d_%H_%M')}_{args.data_name}_{'_'.join(args.mod_list)}_{args.max_eval}_{args.epochs}"

MODE = args.mode # "fwdPass", "depth" or "hybrid", frist for edited forward pass, second for depth

NUM_CHANNELS = args.channels

POLARITY = args.polarity # "both", "sum", "sub", "drop"

SEED = args.seed
OPT = args.opt
