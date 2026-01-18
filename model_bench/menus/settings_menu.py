import os
from pathlib import Path
import yaml
from components.colors import colors
from exp_config import load_cfg, reload_cfg
from components.dataset import Dataset2Class
import exp_config
import questionary

class SettingsMenu:
    def __init__(self):
        self.config_path = Path(__file__).parent.parent / "config.yaml"

    def display_header(self):
        os.system('clear' if os.name == 'posix' else 'cls')
        print(colors.HEADER + "+-----------------------------------+")
        print("|          Settings                 |")
        print("+-----------------------------------+" + colors.ENDC)

    def modify_configuration_menu(self):
        """Menu to modify configuration settings"""
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
                    "12: Batch Size",
                    "13: Back"
                ]
            ).ask()
            
            if choice is None:
                return
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
                self.change_batch_size()
            elif choice_num == "13":
                break

    def show_current_config(self):
        """Display the current configuration settings"""
        cfg = load_cfg()
        print(colors.OKBLUE + "\nCurrent Configuration:" + colors.ENDC)
        print(f"  Name: {cfg.name}")
        print(f"  Dataset: {cfg.dataset}")
        print(f"  Epochs: {cfg.epochs}")
        print(f"  Batch Size: {cfg.batch_size}")
        print(f"  Max Evaluations: {cfg.eval}")
        print(f"  Frames: {cfg.frames}")
        print(f"  Channels: {cfg.channels}")
        print(f"  Polarity: {cfg.polarity}")
        print(f"  Mode: {cfg.mode}")
        print(f"  Seed: {cfg.seed}")
        print(f"  Modules: {cfg.mod_list}")
        print(f"  Optimizer: {cfg.opt}")

    def change_dataset(self):
        """Change dataset configuration"""
        datasets = sorted(Dataset2Class.keys())
        choice = questionary.select(
            "Select dataset:",
            choices=datasets
        ).ask()
        
        if choice:
            self._update_config_field("dataset", choice)
            print(colors.OKGREEN + f"Dataset set to: {choice}" + colors.ENDC)
            questionary.press_any_key_to_continue().ask()

    def change_max_evaluations(self):
        """Change max evaluations configuration"""
        current = load_cfg().eval
        max_eval = questionary.text(
            "Max evaluations:",
            default=str(current)
        ).ask()
        
        if max_eval and max_eval.isdigit():
            self._update_config_field("eval", int(max_eval))
            print(colors.OKGREEN + f"Max evaluations set to: {max_eval}" + colors.ENDC)
            questionary.press_any_key_to_continue().ask()

    def change_epochs(self):
        """Change epochs configuration"""
        current = load_cfg().epochs
        epochs = questionary.text(
            "Number of epochs:",
            default=str(current)
        ).ask()
        
        if epochs and epochs.isdigit():
            self._update_config_field("epochs", int(epochs))
            print(colors.OKGREEN + f"Epochs set to: {epochs}" + colors.ENDC)
            questionary.press_any_key_to_continue().ask()

    def change_batch_size(self):
        """Change batch size configuration"""
        current = load_cfg().batch_size
        batch_size = questionary.text(
            "Batch size:",
            default=str(current)
        ).ask()
        
        if batch_size and batch_size.isdigit():
            self._update_config_field("batch_size", int(batch_size))
            print(colors.OKGREEN + f"Batch size set to: {batch_size}" + colors.ENDC)
            questionary.press_any_key_to_continue().ask()

    def change_frames(self):
        """Change frames configuration"""
        current = load_cfg().frames
        frames = questionary.text(
            "Number of frames:",
            default=str(current)
        ).ask()
        
        if frames and frames.isdigit():
            self._update_config_field("frames", int(frames))
            print(colors.OKGREEN + f"Frames set to: {frames}" + colors.ENDC)
            questionary.press_any_key_to_continue().ask()

    def change_mode(self):
        """Change mode configuration"""
        modes = sorted(exp_config._VALID_MODES)
        choice = questionary.select(
            "Select mode:",
            choices=modes
        ).ask()
        
        if choice:
            self._update_config_field("mode", choice)
            print(colors.OKGREEN + f"Mode set to: {choice}" + colors.ENDC)
            questionary.press_any_key_to_continue().ask()

    def change_channels(self):
        """Change channels configuration"""
        current = load_cfg().channels
        channels = questionary.text(
            "Number of channels:",
            default=str(current)
        ).ask()
        
        if channels and channels.isdigit():
            self._update_config_field("channels", int(channels))
            print(colors.OKGREEN + f"Channels set to: {channels}" + colors.ENDC)
            questionary.press_any_key_to_continue().ask()

    def change_polarity(self):
        """Change polarity configuration"""
        polarities = sorted(exp_config._VALID_POLARITY)
        choice = questionary.select(
            "Select polarity:",
            choices=polarities
        ).ask()
        
        if choice:
            self._update_config_field("polarity", choice)
            print(colors.OKGREEN + f"Polarity set to: {choice}" + colors.ENDC)
            questionary.press_any_key_to_continue().ask()

    def change_modules(self):
        """Change modules configuration"""
        valid_modules = sorted(exp_config._VALID_MODULES)
        current_modules = load_cfg().mod_list
        if not isinstance(current_modules, (list, tuple, set)):
            current_modules = []
        # Build checkbox choices and pre-check currently enabled modules.
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
            print(colors.OKGREEN + f"Modules set to: {', '.join(selected)}" + colors.ENDC)
            questionary.press_any_key_to_continue().ask()

    def change_seed(self):
        """Change seed configuration"""
        current = load_cfg().seed
        seed = questionary.text(
            "Seed:",
            default=str(current)
        ).ask()
        
        if seed and seed.isdigit():
            self._update_config_field("seed", int(seed))
            print(colors.OKGREEN + f"Seed set to: {seed}" + colors.ENDC)
            questionary.press_any_key_to_continue().ask()

    def change_optimizer(self):
        """Change optimizer configuration"""
        optimizers = sorted(exp_config._VALID_OPT)
        choice = questionary.select(
            "Select optimizer:",
            choices=optimizers
        ).ask()
        
        if choice:
            self._update_config_field("opt", choice)
            print(colors.OKGREEN + f"Optimizer set to: {choice}" + colors.ENDC)
            questionary.press_any_key_to_continue().ask()

    def change_experiment_name(self):
        """Change experiment name configuration"""
        current = load_cfg().name
        name = questionary.text(
            "Experiment name:",
            default = str(current)
        ).ask()
        
        if name:
            self._update_config_field("name", name)
            print(colors.OKGREEN + f"Name set to: {name}" + colors.ENDC)
            questionary.press_any_key_to_continue().ask()

    def _update_config_field(self, field: str, value):
        """Update a field in config.yaml and reload"""
        with open(self.config_path, 'r') as f:
            config = yaml.safe_load(f) or {}

        #if YAML was corrupted, reset to empty dict instead of crashing
        if not isinstance(config, dict):
            print(colors.WARNING + f"[Warning] Malformed config, resetting to empty dict." + colors.ENDC)
            config = {}     

        config[field] = value

        with open(self.config_path, 'w') as f:
            yaml.safe_dump(config, f, sort_keys=False)

        exp_config.set_active_config(self.config_path)
        reload_cfg()

    def export_configuration(self):
        """Export current configuration to a YAML file"""
        cfg = load_cfg()

        # Default filename and export path
        default_dir = Path(__file__).parent.parent / "exports" / "configurations"
        default_name = f"config_export_{cfg.name}.yaml"

        dir_path = questionary.path(
            "Select export directory:",
            default=str(default_dir),
            only_directories=True
        ).ask()
        if dir_path is None:
            questionary.press_any_key_to_continue().ask()
            return
        
        while True:
            filename = questionary.text(
                "Enter filename:",
                default=default_name
            ).ask()
            if filename == "":
                print(colors.FAIL + "Filename cannot be empty. Please try again." + colors.ENDC)
                continue
            if filename is None:
                questionary.press_any_key_to_continue().ask()
                return
        
            file_path = Path(filename)

            if '.' in file_path.stem:
                print(colors.FAIL + "Filename cannot contain dots except for the .yaml/.yml extension." + colors.ENDC)
                continue

            filename = file_path.stem + ".yaml"
            break
        try:
            dest_path = Path(dir_path).expanduser().resolve() / filename

            # Warn if file exists
            if dest_path.exists():
                print(colors.WARNING + f"File {dest_path} already exists." + colors.ENDC)
                overwrite = questionary.confirm(
                    "Overwrite?"
                ).ask()
                if not overwrite:
                    questionary.press_any_key_to_continue().ask()
                    return
                
            # Ensure directory exists
            dest_path.parent.mkdir(parents=True, exist_ok=True)

            config_dict = dict(cfg)

            with open(dest_path, 'w') as f:
                yaml.safe_dump(config_dict, f, sort_keys=False, allow_unicode=True)
            print(colors.OKGREEN + f"Config exported successfully to:\n {dest_path}" + colors.ENDC)
            questionary.press_any_key_to_continue().ask()
        
        except Exception as e:
            print(colors.FAIL + f"Error during export: {e}" + colors.ENDC)
            questionary.press_any_key_to_continue().ask()

    def load_config_file(self):
        """Import configuration from a YAML file and overwrite current config.yaml"""
        
        file_path = questionary.path(
            "Insert the path of the configuration file:",
        ).ask()

        if file_path is None:
            questionary.press_any_key_to_continue().ask()
            return
        
        file_path = Path(file_path).expanduser().resolve()

        # Validate file existence and type
        if not file_path.is_file():
            print(colors.FAIL + "Path is not a file" + colors.ENDC)
            questionary.press_any_key_to_continue().ask()
            return
        if not file_path.suffix in ['.yaml', '.yml']:
            print(colors.FAIL + "File must be a .yaml or .yml file" + colors.ENDC)
            questionary.press_any_key_to_continue().ask()
            return
        
        try:
            with open(file_path, 'r') as f:
                new_config = yaml.safe_load(f)

            if not isinstance(new_config, dict):
                raise ValueError("YAML file must contain a configuration dictionary.")
            
            # Validate critical fields before applying
            if "dataset" in new_config and new_config["dataset"] not in Dataset2Class.keys():
                raise ValueError(f"Invalid dataset: {new_config['dataset']}")
            
            if "mod_list" in new_config:
                bad_mods = [m for m in new_config.get("mod_list", []) if m not in exp_config._VALID_MODULES]
                if bad_mods:
                    raise ValueError(f"Invalid modules: {bad_mods}")
            
            if "mode" in new_config and new_config["mode"] not in exp_config._VALID_MODES:
                raise ValueError(f"Invalid mode: {new_config['mode']}")
            
            if "polarity" in new_config and new_config["polarity"] not in exp_config._VALID_POLARITY:
                raise ValueError(f"Invalid polarity: {new_config['polarity']}")
            
            if "opt" in new_config and new_config["opt"] not in exp_config._VALID_OPT:
                raise ValueError(f"Invalid optimizer: {new_config['opt']}")

            for field in ["eval", "epochs", "frames", "channels", "seed", "batch_size"]:
                if field in new_config and not isinstance(new_config[field], int):
                    raise ValueError(f"Field '{field}' must be an integer.")

            # Overwrite local config.yaml
            with open(self.config_path, 'w') as f:
                yaml.safe_dump(new_config, f, sort_keys=False, allow_unicode=True)

            # Update environment variable and reload
            exp_config.set_active_config(self.config_path)
            reload_cfg()
            
            print(colors.OKGREEN + f"Configuration imported successfully from:\n  {file_path}" + colors.ENDC)
            questionary.press_any_key_to_continue().ask()            
        except Exception as e:
            print(colors.FAIL + f"Error during import: {e}" + colors.ENDC)
            questionary.press_any_key_to_continue().ask()            

    def run(self):
        while True:
            self.display_header()
            
            choice = questionary.select(
                "Settings:",
                choices=[
                    "1: View configuration",
                    "2: Modify configuration",
                    "3: Import configuration file",
                    "4: Export configuration file",
                    "5: Back"
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
                self.export_configuration()
            elif choice_num == "5":
                break