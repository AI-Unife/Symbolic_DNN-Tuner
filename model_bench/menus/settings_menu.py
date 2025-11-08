import questionary
import sys
import os
from pathlib import Path
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from components.colors import colors
from exp_config import load_cfg, reload_cfg

class SettingsMenu:
    def __init__(self):
        self.config_path = Path(__file__).parent.parent / "config.yaml"

    def display_header(self):
        os.system('clear' if os.name == 'posix' else 'cls')
        print(colors.HEADER + "+-----------------------------------+")
        print("|          Settings                 |")
        print("+-----------------------------------+" + colors.ENDC)

    def show_current_config(self):
        cfg = load_cfg()
        print(colors.OKBLUE + "\nCurrent Configuration:" + colors.ENDC)
        print(f"  Name: {cfg.name}")
        print(f"  Dataset: {cfg.dataset}")
        print(f"  Epochs: {cfg.epochs}")
        print(f"  Seed: {cfg.seed}")
        print(f"  Modules: {cfg.mod_list}")
        print(f"  Optimizer: {cfg.opt}")
        input("\nPress Enter to continue...")

    def change_dataset(self):
        datasets = [
            "cifar10", 
            "mnist", 
            "fashion-mnist",
            "imagenet16120",
            "gesture"
        ]
        choice = questionary.select(
            "Select dataset:",
            choices=datasets
        ).ask()
        
        if choice:
            self._update_config_field("dataset", choice)
            print(colors.OKGREEN + f"Dataset impostato su: {choice}" + colors.ENDC)
            input("Press Enter to continue...")

    def change_epochs(self):
        current = load_cfg().epochs
        epochs = questionary.text(
            "Number of epochs:",
            default=str(current)
        ).ask()
        
        if epochs and epochs.isdigit():
            self._update_config_field("epochs", int(epochs))
            print(colors.OKGREEN + f"Epochs impostato su: {epochs}" + colors.ENDC)
            input("Press Enter to continue...")

    def change_experiment_name(self):
        current = load_cfg().name
        name = questionary.text(
            "Experiment name:",
            default=current
        ).ask()
        
        if name:
            self._update_config_field("name", name)
            print(colors.OKGREEN + f"Name impostato su: {name}" + colors.ENDC)
            input("Press Enter to continue...")

    def _update_config_field(self, field: str, value):
        """Aggiorna un campo nel config.yaml e ricarica"""
        with open(self.config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        config[field] = value
        
        with open(self.config_path, 'w') as f:
            yaml.safe_dump(config, f, sort_keys=False)
        
        # Ricarica la configurazione
        reload_cfg()

    def reset_to_defaults(self):
        confirm = questionary.confirm(
            "Reset all settings to default?",
            default=False
        ).ask()
        
        if confirm:
            default_config = {
                "eval": 100,
                "epochs": 10,
                "mod_list": [],
                "dataset": "cifar10",
                "name": "model_bench",
                "frames": 16,
                "mode": "fwdPass",
                "channels": 2,
                "polarity": "both",
                "seed": 42,
                "opt": "RS"
            }
            
            with open(self.config_path, 'w') as f:
                yaml.safe_dump(default_config, f, sort_keys=False)
            
            reload_cfg()
            print(colors.OKGREEN + "Configuration reset to defaults!" + colors.ENDC)
            input("Press Enter to continue...")

    def run(self):
        while True:
            self.display_header()
            
            choice = questionary.select(
                "Settings:",
                choices=[
                    "1: Show current configuration",
                    "2: Change dataset",
                    "3: Change epochs",
                    "4: Change experiment name",
                    "5: Reset to defaults",
                    "6: Back to main menu"
                ]
            ).ask()
        
            if not choice:
                break

            choice_num = choice[0]

            if choice_num == "1":
                self.show_current_config()
            elif choice_num == "2":
                self.change_dataset()
            elif choice_num == "3":
                self.change_epochs()
            elif choice_num == "4":
                self.change_experiment_name()
            elif choice_num == "5":
                self.reset_to_defaults()
            elif choice_num == "6":
                break