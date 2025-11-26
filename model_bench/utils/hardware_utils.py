import os
import json
import shutil

from components.colors import colors
import questionary
from utils.model_utils import load_trained_model
from tqdm import tqdm


def load_or_create_nvdla_configs(path="nvdla/nvdla_configs.json"):
    """
    Load or create the nvdla configurations JSON file for hardware configurations.
    :param path: Path to the nvdla configurations JSON file
    :return: List of nvdla configurations
    """
    try: 
        if not os.path.exists(path):
            with open(path, 'w') as f:
                json.dump([], f, indent=4)
        with open(path, 'r') as f:
            return json.load(f)
    except (IOError, json.JSONDecodeError) as e:
        print(colors.FAIL, f"Error loading nvdla configurations: {e}", colors.ENDC)
        return []
    
def hw_visualizzer(hw_mod):
    """Display the available hardware configurations."""

    for config_name, config in hw_mod.nvdla.items():
        print(f"- {config_name} | Path: {config['path']} | Cost: {config['cost']:.2f}")

def add_hw_config():
    """Add a new hardware configuration to the available configurations."""

    nvdla_list = load_or_create_nvdla_configs()
    path=questionary.path("Enter the path of the hardware configuration to add:").ask()

    if path:
        path = os.path.expanduser(path)
        path = os.path.abspath(path)
    if not path or not os.path.exists(path) or not path.endswith(".yaml"):
        print(colors.FAIL, "Invalid path or file does not exist.", colors.ENDC)
        return

    # Copy the .yaml file to nvdla/specs directory
    dest_dir = "nvdla/specs"
    os.makedirs(dest_dir, exist_ok=True)
    base_name = os.path.basename(path)
    dest_path = os.path.join(dest_dir, base_name)
    
    # Handle name conflicts by appending a number to the filename
    if os.path.exists(dest_path):
        name, ext = os.path.splitext(base_name)
        counter = 1
        while True:
            new_name = f"{name}_{counter}{ext}"
            new_dest = os.path.join(dest_dir, new_name)
            if not os.path.exists(new_dest):
                dest_path = new_dest
                base_name = new_name
                break
            counter += 1
    try:
        shutil.copy2(path, dest_path)
    except Exception as e:
        print(colors.FAIL, f"Failed to copy file: {e}", colors.ENDC)
        return

    while True:
        name = questionary.text("Enter the name of the hardware configuration:").ask()
        if any(cfg["name"] == name for cfg in nvdla_list):
            print(colors.FAIL, "The name already exists.", colors.ENDC)
        else:
            break

    while True:
        try:
            area = float(questionary.text("Enter the Area (mm²):").ask())
            break
        except ValueError:
            print(colors.FAIL, "Area must be a numeric value. Please try again.", colors.ENDC)

    while True:
        try:
            cost_par = float(questionary.text("Enter the cost per mm²:").ask())
            break
        except ValueError:
            print(colors.FAIL, "Cost must be a numeric value. Please try again.", colors.ENDC)

    new_config = {
        "name": name,
        "path": base_name,
        "area": area,
        "C/mm2": cost_par
    }

    nvdla_list.append(new_config)

    with open("nvdla/nvdla_configs.json", 'w') as f:
        json.dump(nvdla_list, f, indent=2)
    
    print(colors.OKGREEN,f"Added configuration: {new_config['name']}", colors.ENDC)

def select_hw_config(hw_mod):
    """
    Select hardware configuration(s) from available ones.
    :param hw_mod: Hardware module
    :return: list of selected hardware configuration names
    """

    while True:
        if not hw_mod or not hw_mod.nvdla:
            print(colors.FAIL, "No hardware configurations available.", colors.ENDC)
            return []
        hw_choices = questionary.checkbox(
            "Select the HW configurations:",
            choices=hw_mod.nvdla.keys()
        ).ask()
        if hw_choices == []:
            print(colors.FAIL, "You must select at least one configuration.", colors.ENDC)
        else:
            break
    return hw_choices

def remove_hw_config(hw_mod):
    """
    Remove hardware configuration(s) from available ones.
    :param hw_mod: Hardware module
    :return: list of removed hardware configuration names
    """
    nvdla=load_or_create_nvdla_configs()
    if not nvdla:
        print(colors.FAIL, "No hardware configurations available to remove.", colors.ENDC)
        return
    hw_choices = select_hw_config(hw_mod)

    confirm = questionary.confirm(
        f"Are you sure you want to remove the selected configurations: {', '.join(hw_choices)}?",
        default=False
    ).ask()

    if not confirm:
        print(colors.FAIL, "Operation cancelled by user.", colors.ENDC)
        return
    
    for cfg in nvdla:
        if cfg.get("name") in hw_choices:
            # Attempt to delete the associated file
            file_path = os.path.join("nvdla/specs", cfg.get("path", ""))
            try:
                if os.path.exists(file_path):
                    os.remove(file_path)
            except Exception as e:
                print(colors.FAIL, f"Failed to delete file {file_path}: {e}", colors.ENDC)

    new_list = [cfg for cfg in nvdla if cfg.get("name") not in hw_choices]
    with open("nvdla/nvdla_configs.json", 'w') as f:
        json.dump(new_list, f, indent=2)

    if hw_mod and hasattr(hw_mod, 'nvdla'):
        for name in hw_choices:
            if name in hw_mod.nvdla:
                try:
                    del hw_mod.nvdla[name]
                except KeyError as e:
                    pass

    print(colors.OKGREEN, f"Deleted: {', '.join(hw_choices)}", colors.ENDC)
    return hw_choices