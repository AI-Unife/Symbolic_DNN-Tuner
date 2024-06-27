import importlib
from colors import colors
import os
import numpy as np
from common_interface import common_interface

class module:
    """
    Class for creating and managing loss module instances
    """
    def __init__(self, modules = []):
        self.modules_array = modules
        self.modules_obj = []
        self.modules_name = []
        self.load_modules()

    """
    Creation of module instances
    """
    def load_modules(self):
        for module in self.modules_array:
            try:
                base_dir = "loss." + module
                module_class = getattr(importlib.import_module(base_dir), module)
                if issubclass(module_class, common_interface):
                    self.modules_obj.append(module_class())
                    self.modules_name.append(module)
                else:
                    print(colors.FAIL, f"|  --------- {module} DOESN'T IMPLEMENT INTERFACE  -------  |\n", colors.ENDC)
            except AttributeError:
                print(colors.FAIL, f"|  ----------- {module} CLASS DOESN'T EXIST ----------  |\n", colors.ENDC)
            except ModuleNotFoundError:
                print(colors.FAIL, f"|  ----------- FAILED TO INSTANCIATE {module} ----------  |\n", colors.ENDC)
            except NotImplementedError:
                print(colors.FAIL, f"|  -------------- ERROR IN {module} STRUCTURE -------------  |\n", colors.ENDC)

    """
    Filter rules, actions and problem rules of loaded modules
    :return: Strings containing the set of rules, actions and problem rules of the loaded modules, respectively
    """
    def get_rules(self):      
        rules = ""
        actions = ""
        problems = ""

        for name in self.modules_name:
           module_name = "loss/" + name + ".pl"
           if os.path.exists(module_name):

               rules += "% rules utils in '" + name + "'\n"
               actions += "% action rules in '" + name + "'\n"

               f = open(module_name, 'r')
               lines = f.readlines()
               for line in lines:
                   if "::" in line and ":-" in line:
                       problems += line
                   elif "::" in line:
                       actions += line
                   elif ":-" in line:
                       rules += line
  
               rules += "\n"
               actions += "\n"
               problems += "\n"

               f.close()

        return rules, actions, problems
    
    """
    Update internal state of modules
    """
    def state(self, *args):
        for module in self.modules_obj:
            module.update_state(*args)

    """
    Get values of modules
    :return: list of module values
    """
    def values(self):
        values = []
        for module in self.modules_obj:
            values += module.obtain_values()
        return values

    """
    Calculation of the final value of the loss function
    :return: list of module weights, list of module loss values and final value to be optimised
    """
    def optimiziation(self):
        values = []
        weights = []
        for module in self.modules_obj:
           weights += [module.weight]
           values += [module.optimiziation_function()]

        norm_weights = [w / np.sum(weights) for w in weights]
        final_opt = np.sum([w*v for w,v in zip(norm_weights,values)])

        return weights, values, final_opt

    """
    Printing module values
    """
    def print(self):
        for module in self.modules_obj:
            module.printing_values()

    """
    Plotting graphs of values from each module
    """
    def plot(self):
        for module in self.modules_obj:
            module.plotting_function()

    """
    Saving log informations of each module
    """
    def log(self):
        for module in self.modules_obj:
            module.log_function()