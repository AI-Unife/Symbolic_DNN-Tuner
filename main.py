import datetime
import time
import sys
import os
import copy

os.system("")
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from skopt import gp_minimize
from tensorflow.keras import backend as K

from components.colors import colors
from components.controller import controller
from components.dataset import cifar_data, mnist
from components.gesture_dataset import gesture_data
from components.search_space import search_space
from components.params_checker import paramsChecker
from skopt.space import Real, Integer, Categorical

import config as cfg

# ---------------------------------------------------------------------------------------------------
# SETTING UP EXPERIMENT PARAMETERS
# ---------------------------------------------------------------------------------------------------

print(colors.MAGENTA, "|  ----------- USER PARAMS CONFIGURATION ----------  |\n", colors.ENDC)
print("EXPERIMENT NAME: ", cfg.NAME_EXP)
print("DATASET NAME: ", cfg.DATA_NAME)
print("MAX NET EVAL: ", cfg.MAX_EVAL)
print("EPOCHS FOR TRAINING: ", cfg.EPOCHS)
print("MODULE LIST: ", cfg.MOD_LIST)

# List of directories required for the experiment
required_dirs = ['Model', 'Weights', 'database', 'checkpoints', 'log_folder', 'algorithm_logs', 'dashboard/model', 'symbolic']

# Create missing directories if they do not exist
try:
    os.makedirs(cfg.NAME_EXP, exist_ok=True)
    for folder in required_dirs:
        os.makedirs(f"{cfg.NAME_EXP}/{folder}", exist_ok=True)
except OSError as e:
    print(colors.FAIL, f"|  ----------- FAILED TO CREATE FOLDER: {e} ----------  |\n", colors.ENDC)
    exit()

# Copy symbolic base files
try:
    os.system(f"cp /hpc/home/bzzlca/Symbolic_DNN-Tuner/symbolic_base/* {cfg.NAME_EXP}/symbolic/")
except OSError:
    print(colors.FAIL, "|  ----------- FAILED TO COPY SYMBOLIC DIR ----------  |\n", colors.ENDC)
    exit()

# ---------------------------------------------------------------------------------------------------
# LOADING DATASET
# ---------------------------------------------------------------------------------------------------

if cfg.DATA_NAME == "MNIST":
    X_train, X_test, Y_train, Y_test, n_classes = mnist()
elif cfg.DATA_NAME == "CIFAR-10":
    X_train, X_test, Y_train, Y_test, n_classes = cifar_data()
elif cfg.DATA_NAME == "gesture":
    X_train, X_test, Y_train, Y_test, n_classes = gesture_data()
else:
    print(colors.FAIL, "|  ----------- DATASET NOT FOUND ----------  |\n", colors.ENDC)
    sys.exit()

# ---------------------------------------------------------------------------------------------------
# INITIALIZING SEARCH SPACE AND CONTROLLER
# ---------------------------------------------------------------------------------------------------

dt = datetime.datetime.now()
max_evals = cfg.MAX_EVAL

sp = search_space()
spa = sp.search_sp()
controller = controller(X_train, Y_train, X_test, Y_test, n_classes)

# Define the hyperparameter search space
first_space = copy.deepcopy(spa)
print(colors.MAGENTA, "|  ----------- SEARCH SPACE ----------  |\n", colors.ENDC)
print(first_space)

# Record the start time
start_time = time.time()

# ---------------------------------------------------------------------------------------------------
# OBJECTIVE FUNCTION FOR OPTIMIZATION
# ---------------------------------------------------------------------------------------------------


class obj_wrap():
    """
    Wrapper function for the objective function to be optimized.
    """
    def __init__(self, space):
        self.search_space = space
        
    def objective(self, params):
        """
        this function shows the search space, which is saved in a dedicated log file,
        and calls the function for training the neural network, obtaining the value of the function to be optimised.
        :param params: values of the search space hyperparameters
        :return: value of the objective function to be optimised
        """
        space = {}
        for i, j in zip(self.search_space, params):
            space[i.name] = j
        print("Punto scelto: ", space)
        print("spazio attuale: ")
        print_space(self.search_space)
        f = open("{}/algorithm_logs/hyper-neural.txt".format(cfg.NAME_EXP), "a")
        f.write(str(space) + "\n")
        to_optimize = controller.training(space)
        f.close()
        K.clear_session()
        return to_optimize

# ---------------------------------------------------------------------------------------------------
# CALLBACK FOR STORING BAYESIAN OPTIMIZATION RESULTS
# ---------------------------------------------------------------------------------------------------

class CustomCallback:
    """
    Stores and manages past Bayesian Optimization results.
    """
    def __init__(self):
        self.x_iters = []  
        self.func_vals = []  
        self.last_chenged = -1

    def __call__(self, res):
        """ 
        Saves the last iteration results while handling a rollback mechanism 
        if the function value reaches a predefined threshold (1000).
        """

        # Check if we encounter 1000 for the first time and mark the rollback point
        if self.last_chenged == -1 and res.func_vals[-1] == 1000:
            self.last_chenged = len(self.x_iters)  # Store the rollback index

        # If previously marked rollback point exists and the new function value is not 1000
        if self.last_chenged != -1 and res.func_vals[-1] != 1000:
            # Rollback to the last valid state before the 1000 value was encountered
            self.x_iters = self.x_iters[:self.last_chenged]
            self.func_vals = self.func_vals[:self.last_chenged]
            
            # Reset the rollback marker
            self.last_chenged = -1
            
            # Append the current valid iteration values
            self.x_iters.append(res.x_iters[-1])
            self.func_vals.append(res.func_vals[-1])

        # Ensure we store only new iteration values
        if len(self.x_iters) == 0 or res.x_iters[-1] != self.x_iters[-1]:  
            self.x_iters.append(res.x_iters[-1])
            self.func_vals.append(res.func_vals[-1])
        

# ---------------------------------------------------------------------------------------------------
# ANALYSIS & BAYESIAN OPTIMIZATION PROCESS
# ---------------------------------------------------------------------------------------------------

def print_space(space):
    for i, dim in enumerate(space.dimensions):
        print(f"Dimension {i}: {dim.name} - {dim}")

def start_analysis():
    """
    Runs model analysis and identifies necessary changes in the search space.

    :return: Updated search space and optimization metric
    """
    return controller.diagnosis()

class wrap_constraints(): 
    def __init__(self, space):
        self.space = space
        
    def apply_constraints(self, params):
        """
        Applies constraints to the search space.
        """
        result = True
        # print(search_space)
        for i, dim in enumerate(self.space.dimensions):
            if type(dim) == Categorical:
                if params[i] not in dim.categories:
                    # print("Constraint violated: ", dim.name, params[i])
                    result = False
                    break
            elif type(dim) == Integer or type(dim) == Real:
                if params[i] < dim.low or params[i] > dim.high:
                    # print("Constraint violated: ", dim.name, params[i])
                    result = False
                    break
            else:
                print(f"Type space dimension {dim} - {type(dim)} not valid")
                exit()
        # print("Valid point: ", params)
        return result

def start(new_space, max_iter):
    """
    Performs Bayesian Optimization with search space adaptation.

    :param new_space: Initial hyperparameter search space
    :param max_iter: Number of iterations
    :return: Optimization results
    """
    start_space = copy.deepcopy(new_space)
    print(colors.MAGENTA, "|  ----------- START BAYESIAN OPTIMIZATION ----------  |\n", colors.ENDC)

    custom_callback = CustomCallback()
    controller.set_case(False)
    
    obj_fn = obj_wrap(start_space)
    const_fn = wrap_constraints(start_space)
    apply_constraints = const_fn.apply_constraints
    # First iteration of Bayesian Optimization
    search_res = gp_minimize(obj_fn.objective, start_space, acq_func='EI', n_calls=1, n_random_starts=1,
                             callback=[custom_callback], space_constraint=apply_constraints)

    new_space, to_optimize = start_analysis()
    # new_space = search_space
    print(colors.MAGENTA, "|  ----------- END ITERATION 0 OF BAYESIAN OPTIMIZATION ----------  |\n", colors.ENDC)

    # Iterative Bayesian Optimization
    for opt in range(max_iter):
        print(colors.MAGENTA, f"|  ----------- START ITERATION {opt+1} ----------  |\n", colors.ENDC)

        res = custom_callback
        if len(new_space.dimensions) == len(start_space.dimensions):
            print(colors.WARNING, "-----------------------------------------------------", colors.ENDC)
            print(colors.FAIL, "Continuing Bayesian Optimization", colors.ENDC)
            print(colors.WARNING, "-----------------------------------------------------", colors.ENDC)
            obj_fn = obj_wrap(start_space)
            search_res = gp_minimize(obj_fn.objective, start_space, x0=res.x_iters, y0=res.func_vals, acq_func='EI',
                                     n_calls=1, n_random_starts=0, callback=[custom_callback], space_constraint=apply_constraints)
        else:
            # Update the search space and restart BO
            # search_space = copy.deepcopy(new_space)
            start_space = copy.deepcopy(new_space)
            print(colors.FAIL, "-----------------------------------------------------", colors.ENDC)
            print(colors.WARNING, "Updated search space - Restarting BO", colors.ENDC)
            print(colors.FAIL, "-----------------------------------------------------", colors.ENDC)

            custom_callback = CustomCallback()
            obj_fn = obj_wrap(start_space)
            const_fn = wrap_constraints(start_space)
            apply_constraints = const_fn.apply_constraints
            search_res = gp_minimize(obj_fn.objective, start_space, acq_func='EI',
                                     n_calls=1, n_random_starts=1, callback=[custom_callback], space_constraint=apply_constraints)
        
        # print(len(res.x_iters), len(res.func_vals))
        new_space, to_optimize = start_analysis()
        print_space(new_space)
        print(colors.MAGENTA, f"|  ----------- END ITERATION {opt+1} ----------  |\n", colors.ENDC)


    return search_res

# ---------------------------------------------------------------------------------------------------
# EXECUTE OPTIMIZATION
# ---------------------------------------------------------------------------------------------------

print(colors.OKGREEN, "\nSTARTING ALGORITHM \n", colors.ENDC)
search_res = start(first_space, max_evals)
print(colors.OKGREEN, "\nALGORITHM FINISHED \n", colors.ENDC)

# Logging execution time
print(colors.CYAN, "\nTOTAL TIME --------> \n", time.time() - start_time, colors.ENDC)

# Save results
controller.plotting_obj_function()
controller.save_experience()