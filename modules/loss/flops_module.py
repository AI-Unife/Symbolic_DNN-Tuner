from modules.common_interface import common_interface
from tensorflow.keras.models import load_model
from components.colors import colors
import os
import matplotlib.pyplot as plt
import numpy as np

import flops.flops_calculator as fc
import myconfig as cfg

class flops_module(common_interface):

    #facts and problems for creating the prolog model
    facts = ['flops', 'flops_th', 'nparams', 'nparams_th']
    problems = ['latency', 'model_size']

    #weight of the module for the final loss calculation
    weight = 0.33

    def __init__(self):
        # self.epsilon = 0.33
        self.flops_th = 150000000 # 150 MFLOPs
        self.nparams_th = 2500000 # 2.5M params
        self.tuner_opt_function = []
        self.flops_gap = []
        self.tuner_steps = 0

    def update_state(self, *args):
        self.model = args[0]
        self.flops = args[1]
        self.nparams = args[2]

    def obtain_values(self):
        # has to match the list of facts
        return dict(zip(self.facts, [self.flops, self.flops_th, self.nparams, self.nparams_th]))

    def printing_values(self):
        print("FLOPS: " + str(self.flops))
        print("PARAMS: " + str(self.nparams))

    def optimiziation_function(self, *args):
        # norm flops between 0 - 1
        flops_th = 1
        nflops = self.flops / self.flops_th
        fit_up_flops = flops_th - nflops
        res = -fit_up_flops
        self.flops_gap.append(fit_up_flops)
        self.tuner_steps += 1
        self.tuner_opt_function.append(res)
        return res

    def plotting_function(self):
        x = list(range(self.tuner_steps))
        x = [float(i) for i in x]
        y1 = self.tuner_opt_function
        y2 = self.flops_gap
        plt.plot(x, y1, color='black', label="Total Object Function")
        plt.plot(x, y2, color='blue', label="FLOPS gap")
        plt.savefig("{}/objective_funct.png".format(cfg.NAME_EXP))

    def log_function(self):
        # if os.path.exists("{}/graph_report.txt".format(cfg.NAME_EXP)):
        #     os.remove("{}/graph_report.txt".format(cfg.NAME_EXP))
        f = open("{}/flops_report.txt".format(cfg.NAME_EXP), "a")
        f.write(str(self.flops_th) + " " + str(self.flops) + "\n")
        f.close()
