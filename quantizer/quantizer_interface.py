from abc import ABC, abstractmethod

class quantizer_interface(ABC):
    """
    Common quantizer interface to be implemented by subclasses to build different quantizer.
    The ABC module is used to define an infrastructure for the definition of abstract classes.
    Each method marked with @abstractmethod must be implemented by modules.
    """
            
    @abstractmethod
    def quantizer_function(self):
        """
        Function to quantize the model
        :return: quantized model, score
        """
        pass
    

    @abstractmethod
    def printing_values(self):
        """
        Function to print quantizer values
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

