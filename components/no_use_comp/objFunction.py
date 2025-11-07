<<<<<<<< HEAD:components/objFunction.py
import config as cfg
========
import myconfig as cfg
>>>>>>>> origin/efficient_BO:components/no_use_comp/objFunction.py

class objFunction:
    def __init__(self, search, controller):
        self.search_space = search
        self.controller = controller
        self.space = {}

    def objective(self, params):
        for i,j in zip(self.search_space, params):
            self.space[i.name] = j

        f = open("{}/algorithm_logs/hyperparameters.txt".format(cfg.NAME_EXP), "a")
        to_optimize = self.controller.training(self.space)
        f.close()
        return to_optimize