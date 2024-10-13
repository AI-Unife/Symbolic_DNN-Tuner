from modules.common_interface import common_interface
from tensorflow.keras.models import load_model
from colors import colors
import os
import matplotlib.pyplot as plt
import numpy as np

import flops_calculator as fc

class flops_module(common_interface):

    #facts and problems for creating the prolog model
    facts = ['flops', 'flops_th', 'nparams', 'nparams_th']
    problems = ['latency', 'model_size']

    #weight of the module for the final loss calculation
    weight = 0.05

    def __init__(self):
        self.epsilon = 0.33
        self.flops_th = 1200
        self.nparams_th = 23851784 # inceptionV3 total params
        self.tuner_opt_function = []
        self.flops_gap = []
        self.tuner_steps = 0

    def update_state(self, *args):
        self.accuracy = args[1]
        self.model = args[2]
        self.flops, _ = fc.analyze_model(self.model)
        self.flops = self.flops.total_float_ops
        trainableParams = np.sum([np.prod(v.shape)for v in self.model.trainable_weights])
        nonTrainableParams = np.sum([np.prod(v.shape)for v in self.model.non_trainable_weights])
        self.nparams = trainableParams + nonTrainableParams

    def obtain_values(self):
        # has to match the list of facts
        return dict(zip(self.facts, [self.flops, self.flops_th, self.nparams, self.nparams_th]))

    def printing_values(self):
        print(colors.FAIL, "FLOPS: " + str(self.flops), colors.ENDC)
        print(colors.FAIL, "PARAMS: " + str(self.nparams), colors.ENDC)

    def optimiziation_function(self, *args):
        # norm flops between 0 - 1
        flops_th = 1
        nflops = self.flops / self.flops_th
        fit_up_flops = abs(flops_th - nflops)
        res = -(abs(self.accuracy - fit_up_flops*self.epsilon))
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
        plt.savefig("objective_funct.png")

    def log_function(self):
        if os.path.exists("graph_report.txt"):
            os.remove("graph_report.txt")
        f = open("graph_report.txt", "a")
        f.write(str(self.flops_th) + " " + str(self.flops) + " " + str(self.accuracy) + "\n")
        f.close()
