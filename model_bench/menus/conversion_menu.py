import questionary
import tempfile
import os
import sys
import tensorflow as tf
import torch
import tf2onnx
import onnx
import onnx2torch
import torchinfo


class ConversionMenu:
    def __init__(self):
        pass

    def tf2torch(self):
        while True:
            model = questionary.path("Enter the path of the TensorFlow model:").ask()
            model = os.path.abspath(os.path.expanduser(model))
            if not model or not os.path.exists(model) or not os.path.isfile(model) or not model.endswith(('.keras')):
                print("Invalid model path provided. File must exist and be a .keras file.")
            break

        while True:
            dest_path = questionary.path("Enter the path to save the PyTorch model:").ask()
            if not dest_path:
                print("Invalid destination path provided.")
                continue

            dest_path = os.path.abspath(os.path.expanduser(dest_path or ""))

            if not os.path.exists(dest_path):
                print(f"Directory {dest_path} does not exist.")
                create_dir = questionary.confirm("Do you want to create it?", default=True).ask()
                if create_dir:
                    try:
                        os.makedirs(dest_path)
                    except Exception as e:
                        print(f"Failed to create directory: {e}")
                        continue
            break
        while True:
            name = questionary.text("Enter the name for the saved PyTorch model (without extension):").ask()
            if not name:
                name = os.path.splitext(os.path.basename(model))[0]
                print(f"Using default name: {name}")
            
            save_path = os.path.join(dest_path, f"{name}.pt")
            
            if os.path.exists(save_path):
                overwrite = questionary.confirm(
                    f"File {save_path} already exists. Overwrite?",
                    default=False
                ).ask()
                if not overwrite:
                    continue 
            break
        

        print("Converting model... (this may take a moment)")

        # suppress Tensorflow warnings
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
        import warnings
        warnings.filterwarnings('ignore')

        # Create a temporary directory to save intermediate files
        with tempfile.TemporaryDirectory() as tmpdirname:
            temp_h5 = os.path.join(tmpdirname, "temp_model.h5")
            temp_onnx = os.path.join(tmpdirname, "temp_model.onnx")

            tf_model = tf.keras.models.load_model(model)
            tf_model.save(temp_h5)

            tf_model = tf.keras.models.load_model(temp_h5)

            #redirect stdout
            old_stdout = sys.stdout
            sys.stdout = open(os.devnull, 'w')
            try:
                onnx_model = tf2onnx.convert.from_keras(tf_model, output_path=temp_onnx)
            finally:
                sys.stdout = old_stdout
            
            onnx_model = onnx.load(temp_onnx)
            pytorch_model = onnx2torch.convert(onnx_model)
            
            
            torch.save(pytorch_model, save_path)
            print(f"✓ PyTorch model saved to {save_path}")

            loaded_model = torch.load(save_path, weights_only=False)
            confirm = questionary.confirm("Do you want to display the model summary?", default=True).ask()
            if confirm:
                print("\nModel Summary:")
                print(torchinfo.summary(loaded_model, verbose=0))
            
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