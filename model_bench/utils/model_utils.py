import datetime
import sys
import os
from pathlib import Path

os.system("")
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from tensorflow.keras import backend as K
from tensorflow.keras.models import load_model
import questionary

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from components.colors import colors
from components.dataset import get_datasets
from exp_config import load_cfg


def check_if_path_is_model(path):
    allowed_extensions = (".keras", ".h5")
    if os.path.exists(path) and os.path.isfile(path) and path.endswith(allowed_extensions):
        return True
    else:
        return False

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
        model = load_model(model_path, compile=False)
        return model
    except Exception as e:
        print(colors.FAIL, f"Error loading model from {model_path}: {e}", colors.ENDC)
        return None

def load_trained_model(m_path, show_info=None):
    """Load trained model based on user configuration"""

    cfg = load_cfg()

    if show_info is None:
        show_info = questionary.confirm("Want to see User params configuration and Model summary?", default=False).ask()

    if show_info:
        # cfg.get_experiment_name()
        print(colors.MAGENTA, "|  ----------- USER PARAMS CONFIGURATION ----------  |\n", colors.ENDC)
        print("EXPERIMENT NAME: ", cfg.name)
        print("DATASET NAME: ", cfg.dataset)
        print("MAX NET EVAL: ", cfg.eval)
        print("EPOCHS FOR TRAINING: ", cfg.epochs)
        print("MODULE LIST: ", cfg.mod_list)
        print("mode: ", cfg.mode)

        # iterate over each name in the list of folders and
        # if it doesn't exist, proceed with its creation
    
    # Carica dataset usando il nome normalizzato
    dataset_name = cfg.dataset.strip().lower().replace("-", "")
    
    try:
        if show_info:
            X_train, Y_train, X_test, Y_test, n_classes = get_datasets(dataset_name)
        else:
            X_train, Y_train, X_test, Y_test, n_classes = call_silently(
                get_datasets, dataset_name
            )
    except Exception as e:
        print(colors.FAIL, f"Errore nel caricamento del dataset '{dataset_name}': {e}", colors.ENDC)
        sys.exit(1)

    """ DATASET SECTION -------------------------------------------------------------------------------------------------------- """
    """Load dataset based on user configuration
    if cfg.DATA_NAME == "MNIST":
        # MNIST SECTION --------------------------------------------------------------------------------------------------------
        if show_info:
            X_train, X_test, Y_train, Y_test, n_classes = mnist()
        else:
            X_train, X_test, Y_train, Y_test, n_classes = call_silently(mnist)

    elif cfg.DATA_NAME == "CIFAR-10":
        # CIFAR-10 SECTION -----------------------------------------------------------------------------------------------------
        # obtain images and labels from the cifar dataset
        if show_info:
            X_train, X_test, Y_train, Y_test, n_classes = cifar_data()
        else:
            X_train, X_test, Y_train, Y_test, n_classes = call_silently(cifar_data)
    elif cfg.DATA_NAME == "gesture":
        # GestureDVS128 SECTION -----------------------------------------------------------------------------------------------------
        if show_info:
            X_train, X_test, Y_train, Y_test, n_classes = gesture_data()
        else:
            X_train, X_test, Y_train, Y_test, n_classes = call_silently(gesture_data)
    else:
        print(colors.FAIL, "|  ----------- DATASET NOT FOUND ----------  |\n", colors.ENDC)
        sys.exit()
    """
        
    dt = datetime.datetime.now()
    max_evals = cfg.MAX_EVAL

    # LOADING ALREADY TRAINED MODEL --------------------------------------------------------------------------------------------------------

    model = load_model(m_path, compile=False)
    if show_info:
        model.summary()
        
    return model