
import os
import sys
from pathlib import Path
import tensorflow as tf
import torch
import tf2onnx
import onnx
import onnx2torch
import torchinfo
import tempfile
import warnings
from components.colors import colors
import questionary



class ConversionMenu:
    def __init__(self):
        pass

    def tf2torch(self):
        while True:
            model_path = questionary.path("Select the TensorFlow model file:").ask()
            if model_path is None:
                questionary.press_any_key_to_continue().ask()
                return
            
            model_path = Path(model_path).expanduser().resolve()

            if not model_path.is_file() or not model_path.suffix == '.keras':
                print(colors.FAIL + "Invalid model path provided. File must exist and be a .keras file." + colors.ENDC)
                continue
            break

        while True:
            dir_path = questionary.path(
                "Select the directory to save the PyTorch model:",
                only_directories=True
            ).ask()
            if dir_path is None:
                questionary.press_any_key_to_continue().ask()
                return
            break
            
        while True:
            filename = questionary.text(
                "Enter the name for the saved PyTorch model:",
                default=model_path.stem
            ).ask()
            if filename == "":
                print(colors.FAIL + "Filename cannot be empty. Please try again." + colors.ENDC)
                continue
            if filename is None:
                questionary.press_any_key_to_continue().ask()
                return
        
            file_path = Path(filename)

            if '.' in file_path.stem:
                print(colors.FAIL + "Filename cannot contain dots except for the .pt extension." + colors.ENDC)
                continue

            filename = file_path.stem + ".pt"
            break

        try:
            dest_path = Path(dir_path).expanduser().resolve() / filename

            if dest_path.exists():
                print(colors.WARNING + f"File {dest_path} already exists." + colors.ENDC)
                overwrite = questionary.confirm(
                    "Overwrite?"
                ).ask()
                if not overwrite:
                    questionary.press_any_key_to_continue().ask()
                    return
            
            # Ensure directory exists
            dest_path.parent.mkdir(parents=True, exist_ok=True)

            print("Converting model... (this may take a moment)")

            # suppress Tensorflow warnings
            os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
            warnings.filterwarnings('ignore')

            # Create a temporary directory to save intermediate files
            with tempfile.TemporaryDirectory() as tmpdirname:
                temp_h5 = Path(tmpdirname) / "temp_model.h5"
                temp_onnx = Path(tmpdirname) / "temp_model.onnx"

                tf_model = tf.keras.models.load_model(model_path)
                tf_model.save(temp_h5)

                tf_model = tf.keras.models.load_model(temp_h5)

                #redirect stdout
                old_stdout = sys.stdout
                sys.stdout = open(os.devnull, 'w')
                try:
                    onnx_model = tf2onnx.convert.from_keras(tf_model, output_path=str(temp_onnx))
                finally:
                    sys.stdout = old_stdout
                
                onnx_model = onnx.load(str(temp_onnx))
                pytorch_model = onnx2torch.convert(onnx_model)
                
                
                torch.save(pytorch_model, str(dest_path))
                print(f"✓ PyTorch model saved to {dest_path}")

                loaded_model = torch.load(str(dest_path), weights_only=False)
                confirm = questionary.confirm("Do you want to display the model summary?", default=True).ask()
                if confirm:
                    print("\nModel Summary:")
                    print(torchinfo.summary(loaded_model, verbose=0))
                
                questionary.press_any_key_to_continue().ask()

        except Exception as e:
            print(colors.FAIL + f"Error during conversion: {e}" + colors.ENDC)
            questionary.press_any_key_to_continue().ask()
            

    def run(self):
        while True:
            os.system('clear' if os.name == 'posix' else 'cls')
            choice = questionary.select(
                "Model Conversion Menu - Select an option:",
                choices=[
                    "1: .keras -> .pt (TensorFlow to PyTorch)",
                    "2: Back to Main Menu"
                ]
            ).ask()

            if not choice:
                break

            choice_num = choice[0] if choice else ""

            if choice_num == "1":
                self.tf2torch()
            elif choice_num == "2":
                break