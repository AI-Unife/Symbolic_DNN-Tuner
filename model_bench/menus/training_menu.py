import questionary
import os

from utils.training_utils import fine_tune_model

class TrainingMenu:
    def __init__(self):
        pass

    def start_training(self):
        """Start the training process for a model"""
        model_path = 'modello/best_model.py'
        fine_tune_model(model_path)


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

            choice_num = choice[0] if choice else ""

            if choice_num == "1":
                self.start_training()
            elif choice_num == "2":
                print("Displaying Training Logs...")
            elif choice == "3: Back to Main Menu":
                break