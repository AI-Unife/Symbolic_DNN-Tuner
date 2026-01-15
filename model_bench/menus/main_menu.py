import sys
import os
from pathlib import Path
import questionary
from components.colors import colors
from model_bench.menus.testing_menu import TestingMenu
from model_bench.menus.conversion_menu import ConversionMenu
from model_bench.menus.settings_menu import SettingsMenu
from utils.training_utils import fine_tune_model
from exp_config import load_cfg


class MainMenu:
    def __init__(self):
        self.testing_menu = TestingMenu()
        self.conversion_menu = ConversionMenu()
        self.settings_menu = SettingsMenu()

    def display_header(self):
        os.system('clear' if os.name == 'posix' else 'cls')
        print(colors.HEADER + "+-----------------------------------+")
        print("|            Model Bench            |")
        print("+-----------------------------------+" + colors.ENDC)

    def start_fine_tuning(self):
        """Start the fine-tuning process for a model(s)"""
        self.settings_menu.show_current_config()

        modify = questionary.confirm("Do you want to modify the current configuration before fine-tuning?").ask()
        if modify:
            self.settings_menu.run()
            self.settings_menu.show_current_config()
        
        cfg = load_cfg()

        if not questionary.confirm("Proceed with fine-tuning using this configuration?").ask():
            questionary.press_any_key_to_continue().ask()
            return
        
        while True:
            model_path = questionary.path("Enter the path to the model file:").ask()
            if model_path is None:
                questionary.press_any_key_to_continue().ask()
                return
            
            model_path = Path(model_path).expanduser().resolve()
            if model_path.exists() and model_path.suffix == '.keras':
                break
            else:
                print(colors.FAIL + "Invalid model path or unsupported file type. Please try again." + colors.ENDC)
        
        fine_tune_model(model_path)
        questionary.press_any_key_to_continue().ask()


    def run(self):
        while True:
            self.display_header()
            choice = questionary.select(
                "Select a tool:",
                choices=[
                    "1: Model testing",
                    "2: Model conversion",
                    "3: Model fine-tuning",
                    "4: Settings",
                    "5: Exit"
                ]
            ).ask()
        
            if not choice:
                break

            choice_num = choice[0] if choice else ""
            if choice_num == "1":
                self.testing_menu.run()
            elif choice_num == "2":
                self.conversion_menu.run()
            elif choice_num == "3":
                self.start_fine_tuning()
            elif choice_num == "4":
                self.settings_menu.run()
            elif choice_num == "5":
                print("Goodbye!")
                sys.exit(0)
            else:
                print(colors.FAIL, "Invalid choice. Please try again.", colors.ENDC)