from common_interface import common_interface
from tensorflow.keras.models import load_model
import os
import matplotlib.pyplot as plt

import flops_calculator as fc

class new_flop_calculator(common_interface):

    #facts and problems for creating the prolog model
    facts = ['flops', 'flops_th', 'nparams', 'nparams_th']
    problems = ['latency', 'model_size']

    #weight of the module for the final loss calculation
    weight = 0.76

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
        self.flops = self.obtain_value()

    def obtain_value(self):
        flops, _ = fc.analyze_model(self.model)
        return flops.total_float_ops

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
        plt.savefig("objective_funct_2.png")

    def log_function(self):
        if os.path.exists("graph_report_2.txt"):
            os.remove("graph_report_2.txt")
        f = open("graph_report_2.txt", "a")
        f.write(str(self.flops_th) + " " + str(self.flops) + " " + str(self.accuracy) + "\n")
        f.close()
