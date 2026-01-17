"""
Model utility functions for checking and loading TensorFlow models.
"""
import sys
import os
from pathlib import Path

os.system("")
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from tensorflow.keras.models import load_model
import questionary
from components.colors import colors
from components.dataset import get_datasets
from exp_config import load_cfg


def check_if_path_is_model(path):
    """Check if the given path is a valid model file (.keras or .h5)"""
    path = Path(path)
    return path.is_file() and path.suffix in (".keras", ".h5")


def call_silently(func, *args, **kwargs):
    """Executes a function while temporarily disabling stdout."""
    saved_stdout = sys.stdout
    sys.stdout = open(os.devnull, 'w')  # Disable output
    try:
        return func(*args, **kwargs)
    finally:
        sys.stdout.close()
        sys.stdout = saved_stdout  # Restores output

def load_model_simple(model_path):
    """Load model for analysis (no dataset, no folders)"""
    try:
        model = load_model(str(model_path), compile=False)
        return model
    except Exception as e:
        print(colors.FAIL + f"Error loading model from {model_path}: {e}" + colors.ENDC)
        return None

##### DA CANCELLARE? #####
def load_trained_model(m_path, show_info=None):
    """Load trained model based on user configuration
    use the active config to load the model and dataset
    Args:
        m_path (str): Path to the trained model file (.keras or .h5)
        show_info (bool, optional): Whether to display user params and model summary. 
                                    If None, prompts the user. Defaults to None.
    Returns:
        model: Loaded Keras model
    """

    cfg = load_cfg()

    if show_info is None:
        show_info = questionary.confirm("Want to see User params configuration and Model summary?", default=False).ask()

    if show_info:
        print(colors.OKBLUE + "|  ----------- CONFIGURATION ----------  |\n" + colors.ENDC)
        print("EXPERIMENT NAME: ", cfg.name)
        print("DATASET: ", cfg.dataset)
        print("EPOCHS: ", cfg.epochs)
        print("MODE: ", cfg.mode)
        print("MODULE LIST: ", cfg.mod_list)
        print("MAX EVALUATIONS: ", cfg.eval)
    
    # Carica dataset usando il nome normalizzato
    dataset_name = cfg.dataset.strip().lower().replace("-", "")
    
    try:
        if show_info:
            print(colors.OKBLUE + "\n|  ----------- DATASET SUMMARY ----------  |\n" + colors.ENDC)
            X_train, Y_train, X_test, Y_test, n_classes = get_datasets(dataset_name)
        else:
            X_train, Y_train, X_test, Y_test, n_classes = call_silently(
                get_datasets, dataset_name
            )
    except Exception as e:
        print(colors.FAIL + f"Errore nel caricamento del dataset '{dataset_name}': {e}" + colors.ENDC)
        sys.exit(1)
        
    # LOADING ALREADY TRAINED MODEL --------------------------------------------------------------------------------------------------------
    try:
        model = load_model(str(m_path), compile=False)
        if show_info:
            print(colors.OKBLUE + "\n|  ----------- MODEL SUMMARY ----------  |\n" + colors.ENDC)
            model.summary()
    except Exception as e:
        print(colors.FAIL + f"Errore nel caricamento del modello da '{m_path}': {e}" + colors.ENDC)
        sys.exit(1)

    return model