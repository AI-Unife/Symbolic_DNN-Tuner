import questionary
import os
from modules.loss.hardware_module import hardware_module
from components.colors import colors
from utils.hardware_utils import hw_visualizzer, add_hw_config, remove_hw_config as rm_hw_config

class HardwareMenu:
    def __init__(self, hw_mod=None):
        if hw_mod:
            self.hw_mod = hw_mod
        else:
            self.hw_mod = hardware_module()

    def display_header(self):
        os.system('clear' if os.name == 'posix' else 'cls')
        print(colors.HEADER + "+-----------------------------------+")
        print("|       Hardware Management Menu    |")
        print("+-----------------------------------+" + colors.ENDC)
    
    def view_hardware_configurations(self):
        if not self.hw_mod or not self.hw_mod.nvdla:
            print(colors.WARNING + "No hardware configurations found." + colors.ENDC)
            return
        
        print(colors.OKBLUE + "+----------- AVAILABLE HARDWARE -----------+" + colors.ENDC)  
        hw_visualizzer(self.hw_mod)
        print(colors.OKBLUE + "+------------------------------------------+" + colors.ENDC)

        questionary.press_any_key_to_continue().ask()

    def add_hardware_configuration(self):
        print(colors.OKBLUE + "+----------- ADD NEW HARDWARE -----------+" + colors.ENDC)
        add_hw_config()
        # Refresh hardware module configurations
        self.hw_mod = hardware_module()
        print(colors.OKBLUE + "+------------------------------------------+" + colors.ENDC)
        questionary.press_any_key_to_continue().ask()


    def remove_hardware_configuration(self):
        if not self.hw_mod or not self.hw_mod.nvdla:
            print(colors.FAIL + "No hardware configurations available to remove." + colors.ENDC)
            return
        print(colors.OKBLUE + "+----------- REMOVE HARDWARE -----------+" + colors.ENDC)
        removed = rm_hw_config(self.hw_mod)
        if removed:
            # Refresh hardware module configurations
            self.hw_mod = hardware_module()
        print(colors.OKBLUE + "+------------------------------------------+" + colors.ENDC)
        questionary.press_any_key_to_continue().ask()

    def run(self):
        while True:
            self.display_header()
            choice = questionary.select(
                "Select an option:",
                choices=[
                    "1: View Hardware Configurations",
                    "2: Add New Hardware",
                    "3: Remove Hardware",
                    "4: Back"
                ]
            ).ask()

            if not choice:
                break

            choice_num = choice[0] if choice else ""

            if choice_num == "1":
                self.view_hardware_configurations()
            elif choice_num == "2":
                self.add_hardware_configuration()
            elif choice_num == "3":
                self.remove_hardware_configuration()
            elif choice_num == "4":
                break