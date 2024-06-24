from abc import ABC, abstractmethod

class common_interface(ABC):
    """
    Common interface to be implemented by subclasses to build loss modules
    """

    @classmethod
    def __init_subclass__(cls):
        required_variables = ['facts','problems','weight']
        for var in required_variables:
            if not hasattr(cls, var):
                raise NotImplementedError(f'Required variable `{var}` in {cls} not found')

    """
    Function to update the internal state of the module at each iteration
    """
    @abstractmethod
    def update_state(self):
        pass

    """
    Function to obtain the value of loss parameter
    :return: value of that specific loss parameter
    """
    @abstractmethod
    def obtain_values(self):
        pass

    """
    Function for calculating the loss function value to be minimized
    :return: loss value to be minimized
    """
    @abstractmethod
    def optimiziation_function(self):
        pass

    """
    Function to print module values
    """
    @abstractmethod
    def printing_values(self):
        pass

    """
    Function to plot the results
    """
    @abstractmethod
    def plotting_function(self):
        pass

    """
    Function to save log informations
    """
    @abstractmethod
    def log_function(self):
        pass
