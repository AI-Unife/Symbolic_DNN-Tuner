import importlib
from colors import colors
import os

class module:
    """
    class for creating and managing loss module instances
    """
    def __init__(self, modules):
        self.modules_array = modules
        self.modules_obj = []
        self.modules_name = []

    """
    Creation of module instances
    :return: list of module instances
    """
    def load_modules(self):
        for module in self.modules_array:
            try:
                base_dir = "loss." + module
                self.modules_obj.append(getattr(importlib.import_module(base_dir), module)())
                self.modules_name.append(module)
            except ModuleNotFoundError:
                print(colors.FAIL, "|  ----------- FAILED TO INSTANCIATE MODULE ----------  |\n", colors.ENDC)

        return self.modules_obj

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
               problems += "% problems rules in '" + name + "'\n"

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
           
    def value(self):
        return self.modules_obj[0].obtain_value()

    def optimiziation(self):
        return self.modules_obj[0].optimiziation_function()

    def plot(self):
        self.modules_obj[0].plotting_function()

    def log(self):
        self.modules_obj[0].log_function()