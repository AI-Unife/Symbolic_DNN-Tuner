"""
Hardware utility functions for managing hardware configurations.

This module provides functions to view, add, delete 
and make selections of hardware configurations.
"""
import json
import shutil
import filecmp
from pathlib import Path

from components.colors import colors
import questionary

def load_or_create_nvdla_configs(path="nvdla/nvdla_configs.json"):
    """
    Load or create the nvdla configurations JSON file for hardware configurations.
    :param path: Path to the nvdla configurations JSON file
    :return: List of nvdla configurations
    """
    try: 
        config_path = Path(path)
        if not config_path.exists():
            config_path.parent.mkdir(parents=True, exist_ok=True)
            with open(config_path, 'w') as f:
                json.dump([], f, indent=4)
        with open(config_path, 'r') as f:
            return json.load(f)
    except (IOError, json.JSONDecodeError) as e:
        print(colors.FAIL + f"Error loading nvdla configurations: {e}" + colors.ENDC)
        return []

def hw_visualizzer(hw_mod):
    """
    Display the available hardware configurations.
    :param hw_mod: Hardware module
    """
    if not hw_mod or not hasattr(hw_mod, "nvdla") or not hw_mod.nvdla:
        print(colors.WARNING + "No hardware configurations found.\n" + colors.ENDC)
        return
    try: 
        for config_name, config in hw_mod.nvdla.items():
            path_val = config.get("path", "N/A")
            cost_val = config.get("cost", 0.0)
            print(f"- {config_name} | Path: {path_val} | Cost: {cost_val:.2f}")
    except (AttributeError, KeyError) as e:
        print(colors.FAIL + f"Error displaying hardware configurations: {e}" + colors.ENDC)

def add_hw_config():
    """Add a new hardware configuration to the available configurations."""
    try: 
        nvdla_list = load_or_create_nvdla_configs()
        while True:
            config_path=questionary.path("Enter the path of the hardware configuration to add:").ask()

            # Handle Ctrl+C 
            if config_path is None:
                return

            config_path = Path(config_path).expanduser().resolve()

            if not config_path.exists() or not config_path.suffix == ".yaml":
                print(colors.FAIL + "Invalid path or file does not exist." + colors.ENDC)
                continue
            break

        # Copy the .yaml file to nvdla/specs directory
        dest_dir = Path("nvdla/specs")
        dest_dir.mkdir(parents=True, exist_ok=True)
        base_name = config_path.name
        dest_path = dest_dir / base_name

        # Check if identical file already exists
        identical_file = None
        for existing_file in dest_dir.glob("*.yaml"):
            if filecmp.cmp(str(config_path), str(existing_file), shallow=False):
                identical_file = existing_file.name
                break

        if identical_file:
            # Check if this file is already used by an existing configuration
            existing_config = next((cfg for cfg in nvdla_list if cfg.get("path") == identical_file), None)
            
            if existing_config:
                print(colors.WARNING + f"An identical YAML file is already used by configuration: '{existing_config['name']}'" + colors.ENDC)
                print(colors.WARNING + "You cannot use the same file for multiple configurations." + colors.ENDC)
                
                create_duplicate = questionary.confirm(
                    "Do you want to create a duplicate file with a different name?",
                    default=True
                ).ask()

                if create_duplicate is None:
                    return

                if not create_duplicate:
                    print(colors.WARNING + "Operation cancelled." + colors.ENDC)
                    return
                
                # Create duplicate with new name
                name, ext = config_path.stem, config_path.suffix
                counter = 1
                while True:
                    new_name = f"{name}_{counter}{ext}"
                    new_dest = dest_dir / new_name
                    if not new_dest.exists( ):
                        dest_path = new_dest
                        base_name = new_name
                        break
                    counter += 1
                
                shutil.copy2(str(config_path), str(dest_path))
                print(colors.OKGREEN + f"Successfully copied to {dest_path}" + colors.ENDC)
            else:
                # File exists but not used by any configuration
                print(colors.WARNING + f"An identical YAML file already exists: {identical_file}" + colors.ENDC)
                overwrite = questionary.confirm(
                    f"Do you want to overwrite '{identical_file}'?",
                    default=False
                ).ask()

                if overwrite is None:
                    return

                if not overwrite:
                    # Create with new name
                    name, ext = config_path.stem, config_path.suffix
                    counter = 1
                    while True:
                        new_name = f"{name}_{counter}{ext}"
                        new_dest = dest_dir / new_name
                        if not new_dest.exists():
                            dest_path = new_dest
                            base_name = new_name
                            break
                        counter += 1

                shutil.copy2(str(config_path), str(dest_path))
                print(colors.OKGREEN + f"Successfully copied to {dest_path}" + colors.ENDC)

        else:
            # No identical file, check for name conflict
            if dest_path.exists():
                # Check if this filename is already used
                existing_config = next((cfg for cfg in nvdla_list if cfg.get("path") == base_name), None)
                
                if existing_config:
                    print(colors.WARNING + f"File '{base_name}' is already used by configuration: '{existing_config['name']}'" + colors.ENDC)
                    overwrite = questionary.confirm(
                        f"Do you want to overwrite it? This will affect the existing configuration.",
                        default=False
                    ).ask()
                else:
                    overwrite = questionary.confirm(
                        f"File '{base_name}' already exists. Overwrite?",
                        default=False
                    ).ask()

                if overwrite is None:
                    return

                if not overwrite:
                    # Rename with counter
                    name, ext = config_path.stem, config_path.suffix
                    counter = 1
                    while True:
                        new_name = f"{name}_{counter}{ext}"
                        new_dest = dest_dir / new_name
                        if not new_dest.exists():
                            dest_path = new_dest
                            base_name = new_name
                            break
                        counter += 1

            shutil.copy2(str(config_path), str(dest_path))
            print(colors.OKGREEN + f"Successfully copied to {dest_path}" + colors.ENDC)

        # get configuration name
        while True:
            name = questionary.text("Enter the name of the hardware configuration:").ask()
            if name is None:
                return
            if not name:
                print(colors.FAIL + "Name cannot be empty. Please try again." + colors.ENDC)
                continue
            if any(cfg["name"] == name for cfg in nvdla_list):
                print(colors.FAIL + "The name already exists." + colors.ENDC)
            else:
                break

        #get area
        while True:
            try:
                area_input = questionary.text("Enter the Area (mm²):").ask()
                if area_input is None:
                    return
                if not area_input:
                    raise ValueError("Area cannot be empty.")
                area = float(area_input)
                if area <= 0:
                    raise ValueError("Area must be a positive value.")
                break
            except ValueError as e:
                print(colors.FAIL + f"Invalid area: {e}. Please try again." + colors.ENDC)

        #get cost per mm2
        while True:
            try:
                cost_input = questionary.text("Enter the cost per mm²:").ask()
                if cost_input is None:
                    return
                if not cost_input:
                    raise ValueError("Cost cannot be empty.")
                cost_par = float(cost_input)
                if cost_par <= 0:
                    raise ValueError("Cost must be a positive value.")
                break
            except ValueError as e:
                print(colors.FAIL + f"Invalid cost: {e}. Please try again." + colors.ENDC)

        new_config = {
            "name": name,
            "path": base_name,
            "area": area,
            "C/mm2": cost_par
        }

        nvdla_list.append(new_config)

        with open("nvdla/nvdla_configs.json", 'w') as f:
            json.dump(nvdla_list, f, indent=4)
        
        print(colors.OKGREEN + f"Added configuration: {new_config['name']}" + colors.ENDC)

    except Exception as e:
        print(colors.FAIL + f"Error adding hardware configuration: {e}" + colors.ENDC)

def select_hw_config(hw_mod):
    """
    Select hardware configuration(s) from available ones.
    :param hw_mod: Hardware module
    :return: list of selected hardware configuration names
    """
    try: 
        if not hw_mod or not hw_mod.nvdla:
            print(colors.FAIL + "No hardware configurations available to select." + colors.ENDC)
            return []
        
        while True:
            hw_choices = questionary.checkbox(
                "Select the HW configurations:",
                choices=hw_mod.nvdla.keys()
            ).ask()

            # User pressed Ctrl+C
            if hw_choices is None:
                return []

            if not hw_choices:
                print(colors.FAIL + "You must select at least one configuration." + colors.ENDC)
            else:
                break
        return hw_choices
    except Exception as e:
        print(colors.FAIL + f"Error selecting hardware configuration: {e}" + colors.ENDC)
        return []

def remove_hw_config(hw_mod):
    """
    Remove hardware configuration(s) from available ones.
    :param hw_mod: Hardware module
    :return: list of removed hardware configuration names
    """
    try: 
        nvdla=load_or_create_nvdla_configs()
        if not nvdla:
            print(colors.FAIL + "No hardware configurations available to remove." + colors.ENDC)
            return []
        
        hw_choices = select_hw_config(hw_mod)
        if not hw_choices:
            return []

        confirm = questionary.confirm(
            f"Are you sure you want to remove the selected configurations: {', '.join(hw_choices)}?",
            default=False
        ).ask()

        if not confirm:
            return []
    
        # Remove selected configurations
        specs_dir = Path("nvdla/specs")
        for cfg in nvdla:
            if cfg.get("name") in hw_choices:
                file_path = specs_dir / cfg.get("path", "")
                try:
                    if file_path.exists():
                        file_path.unlink()
                        print(colors.OKGREEN + f"Deleted file: {file_path}" + colors.ENDC)
                except OSError as e:
                    print(colors.FAIL + f"Could not delete file {file_path}: {e}" + colors.ENDC)

        new_list = [cfg for cfg in nvdla if cfg.get("name") not in hw_choices]
        
        with open("nvdla/nvdla_configs.json", 'w') as f:
            json.dump(new_list, f, indent=4)

        # Update hw_mod
        if hw_mod and hasattr(hw_mod, 'nvdla'):
            for name in hw_choices:
                hw_mod.nvdla.pop(name, None)

        print(colors.OKGREEN + f"Removed configurations: {', '.join(hw_choices)}" + colors.ENDC)
        return hw_choices
    
    except Exception as e:
        print(colors.FAIL + f"Error removing hardware configuration: {e}" + colors.ENDC)
        return []