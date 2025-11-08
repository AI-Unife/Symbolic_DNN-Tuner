import questionary
from questionary import Style
import sys
import os
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from components.colors import colors
from model_bench.menus.testing_menu import TestingMenu
from model_bench.menus.conversion_menu import ConversionMenu
from model_bench.menus.training_menu import TrainingMenu
from model_bench.menus.settings_menu import SettingsMenu

class MainMenu:
    def __init__(self):
        self.testing_menu = TestingMenu()
        self.conversion_menu = ConversionMenu()
        self.training_menu = TrainingMenu()
        self.settings_menu = SettingsMenu()

    def display_header(self):
        os.system('clear' if os.name == 'posix' else 'cls')
        print(colors.HEADER + "+-----------------------------------+")
        print("|            Model Bench            |")
        print("+-----------------------------------+" + colors.ENDC)

    def run(self):
        while True:
            self.display_header()
            
            choice = questionary.select(
                "Select a tool:",
                choices=[
                    "1: Model testing",
                    "2: Model conversion",
                    "3: Model training",
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
                self.training_menu.run()
            elif choice_num == "4":
                self.settings_menu.run()
            elif choice_num == "5":
                print("Goodbye!")
                sys.exit(0)
            else:
                print(colors.FAIL, "Invalid choice. Please try again.", colors.ENDC)