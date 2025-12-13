import questionary
import sys
import os
from pathlib import Path
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from components.colors import colors
from exp_config import load_cfg, reload_cfg
import exp_config

class SettingsMenu:
    def __init__(self):
        self.config_path = Path(__file__).parent.parent / "config.yaml"

    def display_header(self):
        os.system('clear' if os.name == 'posix' else 'cls')
        print(colors.HEADER + "+-----------------------------------+")
        print("|          Settings                 |")
        print("+-----------------------------------+" + colors.ENDC)

    def modify_configuration_menu(self):
        while True:
            self.display_header()
            choice = questionary.select(
                "Modify Configuration:",
                choices=[
                    "1: Experment Name",
                    "2: Dataset",
                    "3: Modules",
                    "4: Max Evaluations",
                    "5: Epochs",
                    "6: Frames",
                    "7: Mode",
                    "8: Channels",
                    "9: Polarity",
                    "10: Seed",
                    "11: Optimizer",
                    "12: Back to Settings Menu"
                ]
            ).ask()
            
            if choice is None:
                return

            if not choice:
                break

            choice_num = choice.split(":")[0]

            if choice_num == "1":
                self.change_experiment_name()
            elif choice_num == "2":
                self.change_dataset()
            elif choice_num == "3":
                self.change_modules()
            elif choice_num == "4":
                self.change_max_evaluations()
            elif choice_num == "5":
                self.change_epochs()
            elif choice_num == "6":
                self.change_frames()
            elif choice_num == "7":
                self.change_mode()
            elif choice_num == "8":
                self.change_channels()
            elif choice_num == "9":
                self.change_polarity()
            elif choice_num == "10":
                self.change_seed()
            elif choice_num == "11":
                self.change_optimizer()
            elif choice_num == "12":
                break

    def show_current_config(self):
        cfg = load_cfg()
        print(colors.OKBLUE + "\nCurrent Configuration:" + colors.ENDC)
        print(f"  Name: {cfg.name}")
        print(f"  Dataset: {cfg.dataset}")
        print(f"  Epochs: {cfg.epochs}")
        print(f"  Max Evaluations: {cfg.eval}")
        print(f"  Frames: {cfg.frames}")
        print(f"  Channels: {cfg.channels}")
        print(f"  Polarity: {cfg.polarity}")
        print(f"  Mode: {cfg.mode}")
        print(f"  Seed: {cfg.seed}")
        print(f"  Modules: {cfg.mod_list}")
        print(f"  Optimizer: {cfg.opt}")

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

    def change_max_evaluations(self):
        current = load_cfg().eval
        max_eval = questionary.text(
            "Max evaluations:",
            default=str(current)
        ).ask()
        
        if max_eval and max_eval.isdigit():
            self._update_config_field("eval", int(max_eval))
            print(colors.OKGREEN + f"Max evaluations impostato su: {max_eval}" + colors.ENDC)
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

    def change_frames(self):
        current = load_cfg().frames
        frames = questionary.text(
            "Number of frames:",
            default=str(current)
        ).ask()
        
        if frames and frames.isdigit():
            self._update_config_field("frames", int(frames))
            print(colors.OKGREEN + f"Frames impostato su: {frames}" + colors.ENDC)
            input("Press Enter to continue...")

    def change_mode(self):
        modes = sorted(exp_config._VALID_MODES)
        choice = questionary.select(
            "Select mode:",
            choices=modes
        ).ask()
        
        if choice:
            self._update_config_field("mode", choice)
            print(colors.OKGREEN + f"Mode impostato su: {choice}" + colors.ENDC)
            input("Press Enter to continue...")

    def change_channels(self):
        current = load_cfg().channels
        channels = questionary.text(
            "Number of channels:",
            default=str(current)
        ).ask()
        
        if channels and channels.isdigit():
            self._update_config_field("channels", int(channels))
            print(colors.OKGREEN + f"Channels impostato su: {channels}" + colors.ENDC)
            input("Press Enter to continue...")
    
    def change_polarity(self):
        polarities = sorted(exp_config._VALID_POLARITY)
        choice = questionary.select(
            "Select polarity:",
            choices=polarities
        ).ask()
        
        if choice:
            self._update_config_field("polarity", choice)
            print(colors.OKGREEN + f"Polarity impostato su: {choice}" + colors.ENDC)
            input("Press Enter to continue...")

    def change_modules(self):
        valid_modules = sorted(exp_config._VALID_MODULES)
        current_modules = load_cfg().mod_list
        if not isinstance(current_modules, (list, tuple, set)):
            current_modules = []
        
        choices = [
            questionary.Choice(title=mod, checked=(mod in current_modules))
            for mod in valid_modules
        ]
        
        selected = questionary.checkbox(
            "Select modules:",
            choices=choices
        ).ask()
        
        if selected is not None:
            self._update_config_field("mod_list", selected)
            print(colors.OKGREEN + f"Modules impostati su: {', '.join(selected)}" + colors.ENDC)
            input("Press Enter to continue...")

    def change_seed(self):
        current = load_cfg().seed
        seed = questionary.text(
            "seed:",
            default=str(current)
        ).ask()
        
        if seed and seed.isdigit():
            self._update_config_field("seed", int(seed))
            print(colors.OKGREEN + f"Seed impostato su: {seed}" + colors.ENDC)
            input("Press Enter to continue...")

    def change_optimizer(self):
        optimizers = sorted(exp_config._VALID_OPT)
        choice = questionary.select(
            "Select optimizer:",
            choices=optimizers
        ).ask()
        
        if choice:
            self._update_config_field("opt", choice)
            print(colors.OKGREEN + f"Optimizer impostato su: {choice}" + colors.ENDC)
            input("Press Enter to continue...")

    def change_experiment_name(self):
        current = load_cfg().name
        name = questionary.text(
            "Experiment name:",
            default = str(current)
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

    def load_config_file(self):
        """Load configuration from a user-selected YAML file"""
        file_path = questionary.path(
            "Enter the path to the configuration file:",
            only_files=True
        ).ask()

        if file_path is None:
            return
        
        file_path = Path(file_path).expanduser().resolve()

        if not file_path.is_file() or not file_path.suffix in ['.yaml', '.yml']:
            print(colors.FAIL + "Invalid file path or unsupported file type. Please try again." + colors.ENDC)
            questionary.press_any_key_to_continue().ask()
            return
        
        try:
            with open(file_path, 'r') as f:
                new_config = 


        if file_path and Path(file_path).is_file():
            try:
                # Leggi il contenuto del file selezionato
                with open(file_path, 'r') as f:
                    new_config = yaml.safe_load(f)
                
                # Sovrascrivi il config.yaml corrente
                with open(self.config_path, 'w') as f:
                    yaml.safe_dump(new_config, f, sort_keys=False, allow_unicode=True)
                
                # Aggiorna la variabile d'ambiente e ricarica
                exp_config.set_active_config(self.config_path)
                reload_cfg()
                
                print(colors.OKGREEN + f"Configuration loaded from: {file_path}" + colors.ENDC)
                input("Press Enter to continue...")
            except Exception as e:
                print(colors.FAIL + f"Error loading file: {e}" + colors.ENDC)
                input("Press Enter to continue...")
        else:
            print(colors.FAIL + "Invalid path. Try again." + colors.ENDC)
            input("Press Enter to continue...")
    
    def run(self):
        while True:
            self.display_header()
            
            choice = questionary.select(
                "Settings:",
                choices=[
                    "1: View  configuration",
                    "2: Modify configuration",
                    "3: Load configuration file",
                    "4: Back to main menu"
                ]
            ).ask()
        
            if not choice:
                break

            choice_num = choice[0]

            if choice_num == "1":
                self.show_current_config()
                questionary.press_any_key_to_continue().ask()
            elif choice_num == "2":
                self.modify_configuration_menu()
            elif choice_num == "3":
                self.load_config_file()
            elif choice_num == "4":
                break