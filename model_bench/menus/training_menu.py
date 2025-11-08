import questionary
import os

class TrainingMenu:
    def __init__(self):
        pass

    def run(self):
        while True:
            os.system('clear' if os.name == 'posix' else 'cls')
            choice = questionary.select(
                "Training Menu - Select an option:",
                choices=[
                    "1: Start Training",
                    "2: View Training Logs",
                    "3: Back to Main Menu"
                ]
            ).ask()

            if choice == "1: Start Training":
                print("Starting Model Training...")
            elif choice == "2: View Training Logs":
                print("Displaying Training Logs...")
            elif choice == "3: Back to Main Menu":
                break