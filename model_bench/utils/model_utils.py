"""
Model utility functions for checking and loading TensorFlow models.
"""
import sys
import os
from pathlib import Path

os.system("")
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
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

def load_model_dataset(model_path, show_info=None):
    """ Load trained model and Dataset based on user configuration.
        Uses the active config to load them.
        
    Args:
        model_path (str): Path to the trained model file (.keras or .h5)
        show_info (bool, optional): Whether to display dataset and model summary. 
                                    If None, prompts the user. Defaults to None.
    Returns:
        tuple: A tuple containing:
            - model: Loaded Keras model
            - X_train: Training features
            - Y_train: Training labels
            - X_test: Test features
            - Y_test: Test labels
            - n_classes: Number of classes in the dataset
    """

    cfg = load_cfg()

    if show_info is None:
        show_info = questionary.confirm("Want to see Model and Dataset summary?", default=False).ask()

    try:
        if show_info:
            print(colors.OKBLUE + "\n|  ----------- DATASET SUMMARY ----------  |\n" + colors.ENDC)
            X_train, Y_train, X_test, Y_test, n_classes = get_datasets(cfg.dataset)
        else:
            X_train, Y_train, X_test, Y_test, n_classes = call_silently(
                get_datasets, cfg.dataset
            )
    except Exception as e:
        print(colors.FAIL + f"Error loading dataset '{cfg.dataset}': {e}" + colors.ENDC)
        sys.exit(1)
    
    try:
        model = tf.keras.models.load_model(str(model_path))
        if show_info:
            print(colors.OKBLUE + "\n|  ----------- MODEL SUMMARY ----------  |\n" + colors.ENDC)
            model.summary()
    except Exception as e:
        print(colors.FAIL + f"Error loading model from '{model_path}': {e}" + colors.ENDC)
        sys.exit(1)

    return model, X_train, Y_train, X_test, Y_test, n_classes