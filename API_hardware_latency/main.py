import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import questionary

from API_hardware_latency.custom_hardware_module import custom_hardware_module
from API_hardware_latency.hardware_utils import hw_visualizzer, hw_test_all, hw_choose_specific, add_hw_config
from components.colors import colors
from API_hardware_latency.model_utils import load_trained_model, check_model_path

# Ensure the custom_hardware_module is correctly imported
hw_mod=custom_hardware_module()
hw_names = list(hw_mod.nvdla.keys())

print(colors.CYAN, "|  ----------- HARDWARE LATENCY TESTER ----------  |\n", colors.ENDC)

#Loading of the pre-trained model
while True:
    model_path=questionary.path("Enter the path of the pre-trained model").ask()
    if not model_path:
        model_path = "gesture/dashboard/model/model.keras" #Default path if not provided
        break

    if check_model_path(model_path) == True:
        break
model=load_trained_model(model_path)
#menu for the API
print(colors.CYAN, "|  ----------- HARDWARE LATENCY TESTER ----------  |\n", colors.ENDC)
while True:
    choice = questionary.select(
        "Choose an option:",
        choices=[
            "1: View available hardware configurations",
            "2: Test all hardware configurations",
            "3: Choose specific configurations to test",
            "4: Add a new hardware configuration",
            "5: Exit"],
    ).ask()

    if choice.startswith("1"):
        print(colors.OKBLUE, "|  ----------- AVAILABLE HARDWARE  ----------  |\n", colors.ENDC)  
        hw_visualizzer(hw_names)
        print(colors.OKBLUE, "|  ------------------------------------  |\n", colors.ENDC)
    elif choice.startswith("2"):
        hw_test_all(hw_mod, model)
    elif choice.startswith("3"):
        hw_choose_specific(hw_mod, hw_names, model)
    elif choice.startswith("4"):
        hw_mod=add_hw_config()
        hw_names = list(hw_mod.nvdla.keys())
    elif choice.startswith("5"):
        sys.exit()
    else:
        print(colors.FAIL, "Invalid choice. Please try again.", colors.ENDC)