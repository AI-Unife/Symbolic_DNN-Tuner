import datetime
import time
import sys
import os
import copy

from skopt import gp_minimize, load
from skopt.callbacks import CheckpointSaver
from tensorflow.keras import backend as K

from components.colors import colors
from components.controller import controller
from components.dataset import get_datasets
from components.search_space import search_space
from skopt.space import Real, Integer, Categorical

import config as cfg
from components.random_search import RandomSearch
import numpy as np

def print_experiment_config():
    print(colors.MAGENTA, "|  ----------- USER PARAMS CONFIGURATION ----------  |\n", colors.ENDC)
    print("EXPERIMENT NAME: ", cfg.NAME_EXP)
    print("OPTIMIZATION METHOD: ", cfg.OPT, flush=True)
    print("DATASET NAME: ", cfg.DATA_NAME)
    print("MAX NET EVAL: ", cfg.MAX_EVAL)
    print("EPOCHS FOR TRAINING: ", cfg.EPOCHS)
    print("MODULE LIST: ", cfg.MOD_LIST, flush=True)
    if cfg.DATA_NAME == "gesture":
        print("MODE: ", cfg.MODE, flush=True)
        print("POLARITY: ", cfg.POLARITY, flush=True)
        print("CHANNELS: ", cfg.NUM_CHANNELS, flush=True)
    print("SEED: ", cfg.SEED, flush=True)


def create_experiment_folders():
    required_dirs = ['Model', 'Weights', 'database', 'checkpoints', 'log_folder', 
                     'algorithm_logs', 'dashboard/model', 'symbolic']
    try:
        os.makedirs(cfg.NAME_EXP, exist_ok=True)
        for folder in required_dirs:
            os.makedirs(f"{cfg.NAME_EXP}/{folder}", exist_ok=True)
    except OSError as e:
        print(colors.FAIL, f"Failed to create folder: {e}", colors.ENDC)
        exit()


def copy_symbolic_files():
    try:
        os.system(f"cp ./symbolic_base/* {cfg.NAME_EXP}/symbolic/")
    except OSError:
        print(colors.FAIL, "Failed to copy symbolic directory", colors.ENDC)
        exit()

class ObjectiveWrapper:
    def __init__(self, space, controller):
        self.search_space = space
        self.controller = controller
        
    def objective(self, params):
        space_dict = {dim.name: val for dim, val in zip(self.search_space, params)}
        print("Chosen point:", space_dict)
        print("Actual search space:")
        print_space(self.search_space)

        with open(f"{cfg.NAME_EXP}/algorithm_logs/hyper-neural.txt", "a") as f:
            f.write(str(space_dict) + "\n")

        score = self.controller.training(space_dict)
        K.clear_session()
        print("Score:", score, flush=True)
        return score

def print_space(space):
    for i, dim in enumerate(space.dimensions):
        print(f"Dimension {i}: {dim.name} - {dim}")


def print_diff(old_space, new_space):
    print(colors.WARNING, "Differences in search space:", colors.ENDC)
    for i, (old_dim, new_dim) in enumerate(zip(old_space.dimensions, new_space.dimensions)):
        if old_dim != new_dim:
            print(colors.CYAN, f"Dimension {i}: {old_dim.name} changed from {old_dim} to {new_dim}", colors.ENDC)
        else:
            print(colors.FAIL, f"Dimension {i}: {old_dim.name} unchanged", colors.ENDC)


def apply_constraints(space, params):
    """
    Applies constraints to the search space.
    """
    result = True
    if 'RS' in cfg.OPT:
        return True 
    for i, dim in enumerate(space.dimensions):
        if type(dim) == Categorical:
            if params[i] not in dim.categories:
                result = False
                break
        elif type(dim) == Integer or type(dim) == Real:
            if params[i] < dim.low or params[i] > dim.high:
                result = False
                break
        else:
            print(f"Type space dimension {dim} - {type(dim)} not valid")
            exit()
    return result


def run_optimization(search_space, controller, max_iter):
    all_x, all_y = [], []
    callback = CheckpointSaver("{}/checkpoints/checkpoint.pkl".format(cfg.NAME_EXP), compress=9)
    obj_fn = ObjectiveWrapper(search_space, controller)
    no_rules = ['RS', 'standard']
    with_rules = ['filtered', 'RS_ruled']
    if 'RS' in cfg.OPT:
        random_search = RandomSearch(random_state=cfg.SEED, total_iter=max_iter)
        res = random_search(obj_fn.objective, search_space,  callback=[callback])
    else:
        res = gp_minimize(obj_fn.objective, search_space, acq_func='EI',
                      random_state=cfg.SEED, n_calls=1, n_random_starts=1, callback=[callback])
    
    all_x.extend(res.x_iters)
    all_y.extend(res.func_vals)

    new_space = copy.deepcopy(search_space) if cfg.OPT in no_rules else controller.diagnosis()[0]
    it = 0
    while it < max_iter and not controller.convergence:
        print(colors.MAGENTA, f"--- ITERATION {it+1} ---", colors.ENDC)
        it += 1
        res = load('{}/checkpoints/checkpoint.pkl'.format(cfg.NAME_EXP))
        # print_diff(search_space, new_space)

        if len(new_space.dimensions) == len(search_space.dimensions):
            if cfg.OPT in with_rules:
                for x, y in zip(all_x, all_y):
                    if apply_constraints(new_space, x):
                        if x not in res.x_iters:
                            res.x_iters.append(list(x))          
                            res.func_vals = np.append(res.func_vals, y) 
            x0, y0 = res.x_iters, res.func_vals
                
            obj_fn = ObjectiveWrapper(new_space, controller)
            try:
                if 'RS' in cfg.OPT:
                    res = random_search(obj_fn.objective, search_space, callback=[callback])
                else:
                    res = gp_minimize(obj_fn.objective, new_space, x0=list(x0), y0=list(y0),
                                  acq_func='EI', n_calls=1, n_random_starts=0,
                                  random_state=cfg.SEED, callback=[callback])
            except Exception as e:
                print(colors.FAIL, f"Optimization error: {e}", colors.ENDC)
                if 'RS' in cfg.OPT:
                    res = random_search(obj_fn.objective, search_space, callback=[callback])
                else:
                    res = gp_minimize(obj_fn.objective, new_space, acq_func='EI',
                                  n_calls=1, n_random_starts=1,
                                  random_state=cfg.SEED, callback=[callback])

            if cfg.OPT in with_rules:
                all_x.extend(x for x in res.x_iters if x not in all_x)
                all_y.extend(res.func_vals)
        else:
            search_space = copy.deepcopy(new_space)
            print(colors.WARNING, "Search space changed. Restarting BO...", colors.ENDC)
            obj_fn = ObjectiveWrapper(search_space, controller)
            if 'RS' in cfg.OPT:
                random_search.Xi, random_search.Yi = [], [] # Resetting random search history
                res = random_search(obj_fn.objective, search_space, callback=[callback])
            else:
                res = gp_minimize(obj_fn.objective, new_space, acq_func='EI',
                              n_calls=1, n_random_starts=1,
                              random_state=cfg.SEED,
                              callback=[CheckpointSaver(f"{cfg.NAME_EXP}/checkpoints/checkpoint.pkl", compress=9)])

            if cfg.OPT in with_rules:
                all_x, all_y = list(res.x_iters), list(res.func_vals)

        new_space = copy.deepcopy(search_space) if cfg.OPT in no_rules else controller.diagnosis()[0]

    return res

if __name__ == "__main__":
    os.system("")
    os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

    print_experiment_config()
    create_experiment_folders()
    copy_symbolic_files()


    X_train, Y_train, X_test, Y_test, n_classes = get_datasets(cfg.DATA_NAME.strip().lower().replace("-", ""))

    sp = search_space()
    first_space = sp.search_sp()
    print(colors.MAGENTA, "|  ----------- SEARCH SPACE ----------  |\n", colors.ENDC)
    print(first_space)

    ctrl = controller(X_train, Y_train, X_test, Y_test, n_classes)

    start_time = time.time()
    print(colors.OKGREEN, "\nSTARTING ALGORITHM \n", colors.ENDC)
    res = run_optimization(first_space, ctrl, cfg.MAX_EVAL)
    print(colors.OKGREEN, "\nALGORITHM FINISHED \n", colors.ENDC)
    print(colors.CYAN, "\nTOTAL TIME --------> \n", time.time() - start_time, colors.ENDC)

    ctrl.plotting_obj_function()
    ctrl.save_experience()