import argparse
from datetime import datetime

# def get_experiment_name():
parser = argparse.ArgumentParser()

parser.add_argument("--max_eval", type=int, default=200, help="Max number of evaluations")
parser.add_argument("--epochs", type=int, default=30, help="Epochs for training")
parser.add_argument("--mod_list", nargs="+", default=["accuracy_module"],
                    help="Lista dei moduli separati da spazio (es: hardware_module accuracy_module)")
parser.add_argument("--data_name", type=str, default="gesture", help="Dataset name")
parser.add_argument("--name", type=str, default="gesture", help="Experiment ame")

args = parser.parse_args()


MAX_EVAL = args.max_eval
EPOCHS = args.epochs

for mod in args.mod_list:
    if mod not in ["hardware_module", "accuracy_module", "flops_module"]:
        raise ValueError(f"Module {mod} invalid!, Please choose from hardware_module, accuracy_module, flops_module")

MOD_LIST = args.mod_list # ["hardware_module", "accuracy_module", "flops_module"]

DATA_NAME = args.data_name
NAME_EXP = args.name #f"{datetime.now().strftime('%y_%m_%d_%H_%M')}_{args.data_name}_{'_'.join(args.mod_list)}_{args.max_eval}_{args.epochs}"

MODE = "fwdPass" # "fwdPass" or "depth", frist for edited forward pass, second for depth