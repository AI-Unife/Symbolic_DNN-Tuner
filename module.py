import importlib

class module:
    """
    class for creating and managing loss module instances
    """
    def __init__(self, modules):
        self.modules_array = modules
        self.modules_obj = []

    """
    Creation of module instances
    :return: list of module instances
    """
    def load_modules(self):
        for module in self.modules_array:
            base_dir = "loss." + module
            self.modules_obj.append(getattr(importlib.import_module(base_dir), module)())

        return self.modules_obj

    def get_loaded_modules(self):
        return self.modules_obj

    def value(self):
        return self.modules_obj[0].obtain_value()

    def optimiziation(self):
        return self.modules_obj[0].optimiziation_function()

    def plot(self):
        self.modules_obj[0].plotting_function()

    def log(self):
        self.modules_obj[0].log_function()