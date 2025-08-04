import datetime
import time
import sys
import os
import argparse

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

os.system("")
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from skopt import gp_minimize
from skopt import load
from skopt.callbacks import CheckpointSaver
from tensorflow.keras import backend as K

from components.colors import colors
from components.controller import controller
from components.dataset import cifar_data, mnist
from components.gesture_dataset import gesture_data
from components.search_space import search_space
from components.params_checker import paramsChecker

import config as cfg

from modules.loss.hardware_module import hardware_module
from tensorflow.keras.models import load_model


def check_model_path(path):
    if os.path.isfile(path):
        return True
    else:
        print(colors.FAIL, f"{path}: is not a file or does not exist.", colors.ENDC)

def load_trained_model(m_path):
    # FOLDER SECTION --------------------------------------------------------------------------------------------------------

    # list of folders that can be added to the tuner folder if missing
    new_dir = ['Model', 'Weights', 'database', 'checkpoints', 'log_folder', 'algorithm_logs', 'dashboard/model', 'symbolic']

    # cfg.get_experiment_name()
    print(colors.MAGENTA, "|  ----------- USER PARAMS CONFIGURATION ----------  |\n", colors.ENDC)
    print("EXPERIMENT NAME: ", cfg.NAME_EXP)
    print("DATASET NAME: ", cfg.DATA_NAME)
    print("MAX NET EVAL: ", cfg.MAX_EVAL)
    print("EPOCHS FOR TRAINING: ", cfg.EPOCHS)
    print("MODULE LIST: ", cfg.MOD_LIST)
    # iterate over each name in the list of folders and
    # if it doesn't exist, proceed with its creation
    try:
        if not os.path.exists("{}".format(cfg.NAME_EXP)):
            os.makedirs(cfg.NAME_EXP)
    except OSError:
        print(colors.FAIL, "|  ----------- FAILED TO CREATE FOLDER ----------  |\n", colors.ENDC)
        exit()
    for folder in new_dir:
        try:
            if not os.path.exists("{}/{}".format(cfg.NAME_EXP,folder)):
                os.makedirs("{}/{}".format(cfg.NAME_EXP,folder))
        except OSError:
            print(colors.FAIL, "|  ----------- FAILED TO CREATE FOLDER {} ----------  |\n".format("{}/{}".format(cfg.NAME_EXP,folder)), colors.ENDC)
            exit()
    try:
        os.system("cp symbolic_base/* {}/symbolic/".format(cfg.NAME_EXP))
    except OSError:
        print(colors.FAIL, "|  ----------- FAILED TO COPY SYMBOLIC DIR ----------  |\n".format("{}/{}".format(cfg.NAME_EXP,folder)), colors.ENDC)
        exit()
        1
    if cfg.DATA_NAME == "MNIST":
        # MNIST SECTION --------------------------------------------------------------------------------------------------------
        X_train, X_test, Y_train, Y_test, n_classes = mnist()
    elif cfg.DATA_NAME == "CIFAR-10":
        # CIFAR-10 SECTION -----------------------------------------------------------------------------------------------------

        # obtain images and labels from the cifar dataset
        X_train, X_test, Y_train, Y_test, n_classes = cifar_data()
    elif cfg.DATA_NAME == "gesture":
        # GestureDVS128 SECTION -----------------------------------------------------------------------------------------------------
        X_train, X_test, Y_train, Y_test, n_classes = gesture_data()
    else:
        print(colors.FAIL, "|  ----------- DATASET NOT FOUND ----------  |\n", colors.ENDC)
        sys.exit()
        
    dt = datetime.datetime.now()
    max_evals = cfg.MAX_EVAL

    # LOADING ALREADY TRAINED MODEL --------------------------------------------------------------------------------------------------------


        
    #parser = argparse.ArgumentParser()
    #parser.add_argument("--model", type=model_path, default="gesture/dashboard/model/model.keras", help='insert the path of the model file')
    #args = parser.parse_args()
    #model = load_model(args.model, compile=False)

    check_model_path(m_path)

    model = load_model(m_path, compile=False)
    model.summary()
    return model