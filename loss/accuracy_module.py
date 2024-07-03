from common_interface import common_interface
from colors import colors
import os

class accuracy_module(common_interface):

    #facts and problems for creating the prolog model
    facts = ['new_acc']
    problems = []

    #weight of the module for the final loss calculation
    weight = 0.95

    def __init__(self):
        pass

    def update_state(self, *args):
        self.accuracy = args[1]
        
    def obtain_values(self):
        # has to match the list of facts
        return {'new_acc' : self.accuracy}

    def printing_values(self):
        print(colors.FAIL, "ACCURACY: " + str(self.accuracy), colors.ENDC)

    def optimiziation_function(self, *args):
        return -self.accuracy

    def plotting_function(self):
        pass

    def log_function(self):
        if os.path.exists("acc_report.txt"):
            os.remove("acc_report.txt")
        f = open("acc_report.txt", "a")
        f.write(str(self.accuracy) + "\n")
        f.close()
