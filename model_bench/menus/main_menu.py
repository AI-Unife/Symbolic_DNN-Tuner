import questionary
import sys
import os
from components.colors import colors
from menus.testing_menu import TestingMenu
from menus.conversion_menu import ConversionMenu
from menus.training_menu import TrainingMenu

class MainMenu:
    def __init__(self):
        self.testing_menu = TestingMenu()
        self.conversion_menu = ConversionMenu()
        self.training_menu = TrainingMenu()

    def display_header(self):
        os.system('clear' if os.name == 'posix' else 'cls')
        print(colors.HEADER + "+-----------------------------------+")
        print("|            ModelBench             |")
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
                    "4: Exit"
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
                print("Goodbye!")
                sys.exit(0)
            else:
                print(colors.FAIL, "Invalid choice. Please try again.", colors.ENDC)