import questionary
import os
from modules.loss.hardware_module import hardware_module
from components.colors import colors
from utils.hardware_utils import hw_visualizzer, add_hw_config, remove_hw_config as rm_hw_config

class HardwareMenu:
    def __init__(self, hw_mod=None):
        try:
            self.hw_mod = hw_mod if hw_mod else hardware_module()
        except Exception as e:
            print(colors.FAIL + f"Error initialing hardware module: {e}" + colors.ENDC)
            self.hw_mod = None

    def display_header(self):
        os.system('clear' if os.name == 'posix' else 'cls')
        print(colors.HEADER + "+-----------------------------------+")
        print("|       Hardware Management Menu    |")
        print("+-----------------------------------+" + colors.ENDC)

    def view_hardware_configurations(self):        
        if not self.hw_mod:
            print(colors.FAIL + "Hardware module not initialized." + colors.ENDC)
            questionary.press_any_key_to_continue().ask()
            return

        print(colors.OKBLUE + "+----------- AVAILABLE HARDWARE -----------+" + colors.ENDC)  
        try:
            hw_visualizzer(self.hw_mod)
        except Exception as e:
            print(colors.FAIL + f"Error displaying hardware configurations: {e}" + colors.ENDC)
        print(colors.OKBLUE + "+------------------------------------------+" + colors.ENDC)
        questionary.press_any_key_to_continue().ask()

    def add_hardware_configuration(self):
        print(colors.OKBLUE + "+----------- ADD NEW HARDWARE -----------+" + colors.ENDC)
        try:
            add_hw_config()
            # Refresh hardware module configurations
            self.hw_mod = hardware_module()
        except KeyboardInterrupt:
            print(colors.WARNING + "\nOperation cancelled by user." + colors.ENDC)
        except Exception as e:
            print(colors.FAIL + f"Error adding hardware configuration: {e}" + colors.ENDC)
        print(colors.OKBLUE + "+------------------------------------------+" + colors.ENDC)
        questionary.press_any_key_to_continue().ask()


    def remove_hardware_configuration(self):
        if not self.hw_mod or not hasattr(self.hw_mod, "nvdla") or not self.hw_mod.nvdla:
            print(colors.FAIL + "No hardware configurations available to remove.\n" + colors.ENDC)
            questionary.press_any_key_to_continue().ask()
            return

        print(colors.OKBLUE + "+----------- REMOVE HARDWARE -----------+" + colors.ENDC)
        try:
            removed = rm_hw_config(self.hw_mod)
            if removed:
                # Refresh hardware module configurations
                self.hw_mod = hardware_module()
                print(colors.OKGREEN + f"Successfully removed {len(removed)} configuration(s)." + colors.ENDC)
        except KeyboardInterrupt:
            print(colors.WARNING + "\nOperation cancelled by user." + colors.ENDC)
        except Exception as e:
            print(colors.FAIL + f"Error removing hardware configuration: {e}" + colors.ENDC)
        print(colors.OKBLUE + "+------------------------------------------+" + colors.ENDC)
        questionary.press_any_key_to_continue().ask()

    def run(self):
        try: 
            while True:
                self.display_header()

                try:
                    choice = questionary.select(
                        "Select an option:",
                        choices=[
                            "1: View Hardware Configurations",
                            "2: Add New Hardware",
                            "3: Remove Hardware",
                            "4: Back"
                        ]
                    ).ask()
                except KeyboardInterrupt:
                    print(colors.WARNING + "\nOperation cancelled by user." + colors.ENDC)
                    break

                if not choice:
                    break

                choice_num = choice[0]

                if choice_num == "1":
                    self.view_hardware_configurations()
                elif choice_num == "2":
                    self.add_hardware_configuration()
                elif choice_num == "3":
                    self.remove_hardware_configuration()
                elif choice_num == "4":
                    break
        except KeyboardInterrupt:
            print(colors.WARNING + "\nOperation cancelled by user." + colors.ENDC)
        except Exception as e:
            print(colors.FAIL + f"Unexpected error in menu: {e}" + colors.ENDC)