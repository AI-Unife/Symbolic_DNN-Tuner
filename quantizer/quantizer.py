
from quantizer.quantizer_interface import quantizer_interface

class quantizer_module(quantizer_interface):
    
    def __init__(self): 
        # Initialize any necessary attributes here
        pass  

    ## Add useful functions for your quantizer here
    
    
    def quantize_function(self, model):
        # Insert here the quantization function
        model = model
        quantized_model = model  # Replace with actual quantization logic
        score = quantized_model.evaluate(model)
        return model, score

    def printing_values(self):
        # Insert here the code to print quantizer values if you have any
        pass

    def plotting_function(self):
        # Insert here the code to plot quantizer results if you have any
        pass

    def log_function(self):
        string_to_write = "" # insert here the string to write if you want to log something
        f = open("{}/algorithm_logs/quantizer_report.txt".format(self.cfg.name), "a")
        f.write(string_to_write + "\n")
        f.close()
