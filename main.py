import datetime
import time
import sys
import os

os.system("")
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from skopt import gp_minimize
from skopt import load
from skopt.callbacks import CheckpointSaver
from tensorflow.keras import backend as K

from colors import colors
from controller import controller
from dataset import cifar_data, mnist
from search_space import search_space
from params_checker import paramsChecker

# FOLDER SECTION --------------------------------------------------------------------------------------------------------

# list of folders that can be added to the tuner folder if missing
new_dir = ['Model', 'Weights', 'database', 'checkpoints', 'log_folder', 'algorithm_logs', 'dashboard/model']

# iterate over each name in the list of folders and
# if it doesn't exist, proceed with its creation
for folder in new_dir:
    try:
        if not os.path.exists(folder):
            os.makedirs(folder)
    except OSError:
        print(colors.FAIL, "|  ----------- FAILED TO CREATE FOLDER ----------  |\n", colors.ENDC)

# MNIST SECTION --------------------------------------------------------------------------------------------------------

# X_train, X_test, Y_train, Y_test, n_classes = mnist()
# CIFAR-10 SECTION -----------------------------------------------------------------------------------------------------

# obtain images and labels from the cifar dataset
X_train, X_test, Y_train, Y_test, n_classes = cifar_data()
dt = datetime.datetime.now()
max_evals = 0

X_train = X_train[:len(X_train)//100]
X_test = X_test[:len(X_test)//100]
Y_train = Y_train[:len(Y_train)//100]
Y_test = Y_test[:len(Y_test)//100]

# define the hyper-parameters search space, instantiating its class 
# also instantiates the controller class
sp = search_space()
spa = sp.search_sp()
controller = controller(X_train, Y_train, X_test, Y_test, n_classes)

# initialises search space to empty, the set of hyperparameters will be the objective function
space = {}
start_time = time.time()


def update_space(new_space):
    """
    Updates the search space with a new one
    :param new_space: new search space
    :return: updated search space
    """
    global search_space
    search_space = new_space
    return search_space

# updates the search space with that of the initial hyperparameters
search_space = update_space(spa)


def objective(params):
    """
    this function shows the search space, which is saved in a dedicated log file,
    and calls the function for training the neural network, obtaining the value of the function to be optimised.
    :param params: values of the search space hyperparameters
    :return: value of the objective function to be optimised
    """
    space = {}
    for i, j in zip(search_space, params):
        space[i.name] = j
    print(space)
    f = open("algorithm_logs/hyper-neural.txt", "a")
    f.write(str(space) + "\n")
    to_optimize = controller.training(space)
    f.close()
    K.clear_session()
    return to_optimize


def start_analisys():
    """
    function that calls the search and resolution of anomalies affecting the neural network.
    :return: new search space and the value to be optimised (negative accuracy value)
    """
    new_space, to_optimize = controller.diagnosis()
    return new_space, to_optimize


def check_continuing_BO(new_space, x_iters, func_vals):
    func_vals = func_vals.tolist()
    for x in x_iters:
        for n, i in zip(new_space, x):
            if i < n.low or i > n.high:
                _ = func_vals.pop(x_iters.index(x))
                x_iters.remove(x)
                break
    return x_iters, func_vals


def start(search_space, iter):
    """
    Starting bayesian Optimization
    :param iter: iterations to be executed by the tuner
    :return: research result
    """
    print(colors.MAGENTA, "|  ----------- START BAYESIAN OPTIMIZATION ----------  |\n", colors.ENDC)

    checkpoint_saver = CheckpointSaver("checkpoints/checkpoint.pkl", compress=9)
    # optimization of the objective function
    controller.set_case(False)
    search_res = gp_minimize(objective, search_space, acq_func='EI', n_calls=1, n_random_starts=1,
                             callback=[checkpoint_saver])

    # K.clear_session()
    # start neural network analysis and problem fixing
    new_space, to_optimize = start_analisys()

    # execute 'iter' iterations to optimise the network
    for opt in range(iter):
        # restore checkpoint
        if len(new_space) == len(search_space):
            # controller.set_case(True)
            res = load('checkpoints/checkpoint.pkl')

            # if the new search space has the same length as the previous one,
            # proceed with the same bayesian optimization
            try:
                # print(new_space)
                search_res = gp_minimize(objective, new_space, x0=res.x_iters, y0=res.func_vals, acq_func='EI',
                                         n_calls=1,
                                         n_random_starts=0, callback=[checkpoint_saver])
                print(colors.WARNING, "-----------------------------------------------------", colors.ENDC)
                print(colors.FAIL, "Inside BO", colors.ENDC)
                print(colors.WARNING, "-----------------------------------------------------", colors.ENDC)
            except:
                print(new_space)
                #res.x_iters, res.func_vals = check_continuing_BO(new_space, res.x_iters, res.func_vals)
                search_res = gp_minimize(objective, new_space, y0=res.func_vals, acq_func='EI',
                                         n_calls=1,
                                         n_random_starts=1, callback=[checkpoint_saver])
                print(colors.FAIL, "-----------------------------------------------------", colors.ENDC)
                print(colors.WARNING, "Other BO", colors.ENDC)
                print(colors.FAIL, "-----------------------------------------------------", colors.ENDC)
        else:
            # otherwise perform a new optimization, updating the search space of the hyperparameters
            search_space = update_space(new_space)
            search_res = gp_minimize(objective, new_space, acq_func='EI', n_calls=1, n_random_starts=1,
                                     callback=[checkpoint_saver])

        # start neural network analysis and problem fixing
        new_space, to_optimize = start_analisys()

    return search_res

# starts the training and tuning process of the neural network
print(colors.OKGREEN, "\nSTART ALGORITHM \n", colors.ENDC)
search_res = start(search_space, max_evals)
print(search_res)
print(colors.OKGREEN, "\nEND ALGORITHM \n", colors.ENDC)
end_time = time.time()

print(colors.CYAN, "\nTIME --------> \n", end_time - start_time, colors.ENDC)

# generates graphs of loaded modules, also saves training progress on an a DB
controller.plotting_obj_function()
controller.save_experience()