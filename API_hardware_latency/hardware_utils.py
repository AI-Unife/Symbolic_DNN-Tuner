import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import shutil
from API_hardware_latency.custom_hardware_module import custom_hardware_module
from components.colors import colors
import questionary

#list of available hardware
def hw_visualizzer(hw_names):
    print("Available hardware configurations:")
    for idx, name in enumerate(hw_names,1):
        print(f"{idx}: {name}")

#display available hardware configurations
def hw_test_all(hw_mod, model):
    print(colors.OKBLUE, "|  ----------- HARDWARE TESTED ----------  |\n", colors.ENDC)
    for config_name, config in hw_mod.nvdla.items():
        latency = hw_mod.get_model_latency(model, config['path'])
        print(f"Configuration: {config_name} | Latency: {latency/(10**9):.6f} secondi")
    print(colors.OKBLUE, "|  ------------------------------------  |\n", colors.ENDC)


#choose of the specific hardwares configurations to test
def hw_choose_specific(hw_mod, hw_names, model):
    print(colors.OKGREEN, "|  ----------- CHOOSE CONFIGURATIONS TO TEST ----------  |\n", colors.ENDC)
    hw_choose=questionary.checkbox(
        "Select the configurations to test:", 
        choices=hw_names
    ).ask()

    print(colors.OKBLUE, "|  ----------- HARDWARE TESTED ----------  |\n", colors.ENDC)
    for hw_c in hw_choose:
        if hw_c in hw_mod.nvdla:
            latency = hw_mod.get_model_latency(model, hw_mod.nvdla[hw_c]['path'])
            print(f"Configuration: {hw_c} | Latency: {latency/10**9:.6f} secondi")
    print(colors.OKBLUE, "|  ------------------------------------  |\n", colors.ENDC)


#Add a new configuration to the available configurations
def add_hw_config():
    source_path=questionary.path("Enter the path of the hardware configuration to add").ask()
    if not source_path or not os.path.exists(source_path):
        print(colors.FAIL, "Invalid path or file does not exist.", colors.ENDC)
        return
    destination_folder = "nvdla/specs"
    destination_path = os.path.join(destination_folder, os.path.basename(source_path))

    shutil.copy(source_path,destination_path)
    hw_mod=custom_hardware_module()
    return hw_mod
