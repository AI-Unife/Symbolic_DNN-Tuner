from abc import ABC, abstractmethod

class common_interface(ABC):
    """
    Common interface to be implemented by subclasses to build loss modules.
    The ABC module is used to define an infrastructure for the definition of abstract classes.
    Each method marked with @abstractmethod must be implemented by modules.
    """

    @classmethod
    def __init_subclass__(cls):
        """
        This method is called when a class implementing this interface is instantiated
        This is used to check that the necessary attributes are present in each instance
        """
        required_variables = ['facts','problems','weight']
        for var in required_variables:
            if not hasattr(cls, var):
                raise NotImplementedError(f"Required variable {var} in {cls} not found")

    @abstractmethod
    def update_state(self):
        """
        Function to update the internal state of the module at each iteration
        """
        pass

    @abstractmethod
    def obtain_values(self):
        """
        Function to obtain the value of loss parameter
        :return: value of that specific loss parameter
        """
        pass

    @abstractmethod
    def optimiziation_function(self):
        """
        Function for calculating the loss function value to be minimized
        :return: loss value to be minimized
        """
        pass

    @abstractmethod
    def printing_values(self):
        """
        Function to print module values
        """
        pass

    @abstractmethod
    def plotting_function(self):
        """
        Function to plot the results
        """
        pass

    @abstractmethod
    def log_function(self):
        """
        Function to save log informations
        """
        pass
