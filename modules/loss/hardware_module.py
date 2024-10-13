from modules.common_interface import common_interface
from colors import colors
import os

import flops_calculator as fc
from pathlib import Path
from tensorflow.keras import layers, models

import nvdla.profiler as profiler
from dynamic_net import dynamic_net

from math import sqrt

class hardware_module(common_interface):

    #facts and problems for creating the prolog model
    facts = ['hw_latency', 'lenet_latency']
    problems = ['out_range']
    
    #weight of the module for the final loss calculation
    weight = 0

    def __init__(self):
        # cost value per square millimeter, 10K / mm2
        self.cost_par = 10000
        # attribute indicating how much cost weighs against latency value
        self.weight_cost = 0.75
        
        nvdla_list = [{'name': "nv_small", 'path': "nv_small64_fp32.yaml", 'area': 2.824},
                      {'name': "nv_small256", 'path': "nv_small256_fp32.yaml", 'area': 3.091},
                      {'name': "nv_large", 'path': "nv_large2048_fp32.yaml", 'area': 3.809}]
                      
        # init list of available configurations to an empty dict
        self.nvdla = {}
        
        # iterate over each configuration
        for config in nvdla_list:
            if os.path.exists('nvdla/specs/' + config['path']):
                self.nvdla[config['name']] = {'path': config['path'],
                                              'cost': round(self.cost_par*config['area'], 2),
                                              'latency': 0,
                                              'total_cost': 0}
            else:
                print(colors.FAIL, f"|  --------- {config['name']} CONFIGURATION FILE DOESN'T EXIST  -------  |\n", colors.ENDC)
        
        if self.nvdla == {}:
            raise ModuleNotFoundError("No NVDLA configuration found")

        # maximum cost and latency values
        self.max_latency = 20000000
        self.last_flops = 0
        self.nvdla  = dict(sorted(self.nvdla.items(), key=lambda item: item[1]['cost'], reverse=True))
        first_el = next(iter(self.nvdla))
        self.max_cost = self.nvdla[first_el]['cost']
       
    def update_state(self, *args):
        # import current model reference
        self.model = args[2]

        self.flops, _ = fc.analyze_model(self.model)
        self.flops = self.flops.total_float_ops
        
        if self.last_flops == self.flops:
            return

        # for each configuration calculate the latency and the total cost
        for config_key in self.nvdla:
            config_path = self.nvdla[config_key]['path']
            self.nvdla[config_key]['latency'] = self.get_model_latency(self.model, config_path)
            latency_temp = self.nvdla[config_key]['latency'] / self.max_latency
            cost_temp = self.nvdla[config_key]['cost'] / self.max_cost
            self.nvdla[config_key]['total_cost'] = round((cost_temp * self.weight_cost) + (latency_temp * (1-self.weight_cost)), 4)
        
        # sort the configurations by cost
        # this will be useful to determine the optimal configuration
        sorted_config = dict(sorted(self.nvdla.items(), key=lambda item: item[1]['total_cost']))
        self.nvdla = sorted_config
        first_el = next(iter(self.nvdla))
        self.latency = self.nvdla[first_el]['latency']
        self.cost = self.nvdla[first_el]['cost']
        self.total_cost = self.nvdla[first_el]['total_cost']
        self.current_config = first_el
        self.leNet_latency = self.get_model_latency(self.LENET(), config_path)
        
        self.last_flops = self.flops
       
    def obtain_values(self):
        # has to match the list of facts
        return {'hw_latency' : self.latency, 'lenet_latency' : self.leNet_latency}

    def printing_values(self):
        print(colors.FAIL, f"LATENCY: {self.latency} [{self.current_config}]", colors.ENDC)
        print(colors.FAIL, f"COST: {self.cost}$", colors.ENDC)
        print(colors.FAIL, f"TOTAL COST: {self.total_cost}", colors.ENDC)
        print(colors.FAIL, f"LENET: {self.leNet_latency}, {sqrt(self.latency) / sqrt(self.leNet_latency)}", colors.ENDC)
   
    def optimiziation_function(self, *args):
        return -self.total_cost

    def plotting_function(self):
        pass

    def log_function(self):
        pass

    def LENET(self):
        """
        test method to build LeNet
        :return: LeNet model
        """
        model = models.Sequential()
        model.add(layers.Conv2D(6, 5, activation='tanh', padding="same", input_shape=(28, 28, 1))) 
        model.add(layers.AveragePooling2D(2))
        model.add(layers.Activation('sigmoid'))
        model.add(layers.Conv2D(16, 5, activation='tanh'))
        model.add(layers.AveragePooling2D(2))
        model.add(layers.Activation('sigmoid'))
        #model.add(layers.Conv2D(120, 5, activation='tanh'))
        model.add(layers.Flatten())
        model.add(layers.Dense(120, activation='tanh'))
        model.add(layers.Dense(84, activation='tanh'))
        model.add(layers.Dense(10, activation='softmax'))
        return model

    def get_model_latency(self, model, config_path):
        """
        Method that based on calls to a profiler calculates the total latency of a network
        :param model: model to be evaluated
        :return: model latency
        """
        # counter to accumulate the latencies of each network layer
        total_latency = 0
        
        # initialize all the variables that will be passed to the profiler and that characterize the convolutional layers
        in_ch, out_ch, kernel, stride, padding, batch, bias = 0, 0, 0, 0, 2, 1, False

        # define the configuration on which to perform the evaluation and the log file name in which to save the informations
        nvdla_path = config_path
        log_file = "profiler_logs.txt"
        
        # build the path where the configuration file is located
        work_p = os.getcwd()
        config_p = Path(work_p).joinpath('nvdla').joinpath('specs').joinpath(nvdla_path)
        nvdla = profiler.nvdla(config_p)
        
        # build a list containing input dimensions, following the format of pytorch on which the profiler is based
        # the list contains this informations: [number of batches, number of channels, height, width]
        input_size = model.layers[0].output.shape
        input_size = [batch, input_size[3], input_size[1], input_size[2]]

        #model = self.LENET()
        #input_size = [batch, 1, 28, 28]
        
        # iterate over each layer of the model
        for i in model.layers:
            layer_class = i.__class__.__name__

            # if the current layer is a convolution
            if layer_class in ['Conv2D']:
                
                # build the list containing the output dimensions of the convolutional layer
                out_size = i.output.shape
                out_size = [batch, out_size[3], out_size[1], out_size[2]]

                # set the convolutional variables to pass to the profiler with values from the layer
                in_ch = i.input.shape[-1]
                out_ch = i.output.shape[-1]
                kernel = i.kernel_size[0]
                stride = i.strides[0]
                bias = i.use_bias

                # set the padding value according to the padding type of the layer
                input_size = i.input.shape
                input_size = [batch, input_size[3], input_size[1], input_size[2]]
                if i.padding == 'valid':
                    padding = 0
                else:
                    padding = int((kernel - 1) / 2)

                # build the profiler object that maps the convolutional layer and get its latency value
                conv_obj = profiler.Conv2d(nvdla, log_file, i.name, out_size, in_ch, out_ch, kernel, stride, padding, 1, bias)
                total_latency += conv_obj.forward(input_size)
                
            elif layer_class in ['Dense']:
                # if the layer is dense, get the input/output features information and
                # save in the variables that will be passed to the profiler
                in_f = i.input.shape[-1]
                out_f = i.output.shape[-1]
                bias = i.use_bias
                
                # build the list containing the output dimensions of the dense layer
                out_size = [batch, out_f]
                
                # build the profiler object that maps the dense layer and get its latency value
                dense_obj = profiler.Linear(nvdla, log_file, i.name, out_size, in_f, out_f, bias)
                total_latency += dense_obj.forward([batch, in_f])

        return total_latency

