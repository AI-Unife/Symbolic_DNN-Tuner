from common_interface import common_interface

class new_flop_calculator(common_interface):

    #facts and problems for creating the prolog model
    facts = ['flops', 'flops_th', 'nparams', 'nparams_th']
    problems = ['latency', 'model_size']

    #weight of the module for the final loss calculation
    weight = 1

    def __init__(self):
        self.flops_th = 1200
        self.nparams_th = 23851784 # inceptionV3 total params
        self.tuner_opt_function = []
        self.flops_gap = []
        self.tuner_steps = 0
    
    def obtain_value(self):
        flops, _ = fc.analyze_model(model)
        return flops.total_float_ops

    def optimiziation_function(self):
        # norm flops between 0 - 1
        flops_th = 1
        nflops = flops / self.flops_th
        fit_up_flops = abs(flops_th - nflops)
        res = -(abs(accuracy - fit_up_flops*self.epsilon))
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
        f.write(str(self.flops_th) + " " + str(self.flops) + " " + str(self.score[1]) + "\n")
        f.close()
