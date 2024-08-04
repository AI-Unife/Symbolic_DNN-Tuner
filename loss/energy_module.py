from common_interface import common_interface
from colors import colors
import os

import flops_calculator as fc
from pathlib import Path
from tensorflow.keras import layers, models

import profiler

class energy_module(common_interface):
    """
    This module makes it possible to identify the best available hardware architecture,
    in terms of flops and watts, for a given deep neural network.
    When there is no architecture powerful enough to contain the neural network, an attempt is made to decrease its size.
    """
    #facts and problems for creating the prolog model
    facts = ['low_p', 'high_p', 'flops_w', 'last_flops', 'new_latency']
    problems = []

    #weight of the module for the final loss calculation
    weight = 0

    def __init__(self):
        # each config has a name, a flops limit and a list of components
        # each fpga component will have a consumption value expressed in watt
        config_list = [{'name': 'config_1', 'limit' : 45000000, 'mem_1' : 1.5, 'cpu_1': 1.1, 'mem_2' : 0.6},
                       {'name': 'config_2', 'limit' : 20000000, 'mem_1' : 2.4, 'cpu_1': 0.3},
                       {'name': 'config_3', 'limit' : 30000000, 'mem_1' : 1.7, 'cpu_1': 1.8, 'mem_2' : 0.1}]

        # for each configuration save the power and flops/watt
        self.fpga_models = {}

        # iterate over each configuration
        for config in config_list:
            # calculate the power consumed by each configuration as
            # the sum of all contributions from its components
            # initially, set the accumulator to 0
            power = 0

            # iterate over each component
            for component in config.keys():

                # not consider name and flops limit data for power calculation
                if component != 'name' and component != 'limit':
                    power += config[component]

            # if the power value of the current configuration has not been saved, 
            # add it to the vocabulary with the limit value
            config_name = config['name']
            if config_name not in self.fpga_models.keys():
                self.fpga_models[config_name] = {'power_c': {}, 'limit': {}}

            # flops limit and power associated with the configuration name
            self.fpga_models[config_name]['power_c'] = power
            self.fpga_models[config_name]['limit'] = (config['limit'] / 10**6) / power

        # sort the configurations by flops limit
        # this will be useful to determine the optimal configuration
        sorted_config = dict(sorted(self.fpga_models.items(), key=lambda item: item[1]['limit']))

        # set the initial configuration with the first in the list
        # save the name of the current configuration and the flops range
        # the range over which it operates is given by a lower value, in this case 0,
        # and the upper flop limit, the limit of the current configuration
        first_el = next(iter(sorted_config))
        self.current_config = first_el
        self.current_range = [0, sorted_config[first_el]['limit']]

        self.fpga_models = sorted_config
        self.last_flops = 0

    def update_state(self, *args):
        # at each iteration update the number of flops of the current network,
        # as well as its consumption on the current hardware architecture
        self.power = self.fpga_models[self.current_config]['power_c']
        self.model = args[2]
        self.flops, _ = fc.analyze_model(self.model)
        self.flops = self.flops.total_float_ops
        self.flops_w = (self.flops / 10**6) / self.power
        self.latency = self.get_model_latency(self.model)

    def fix_configuration(self):
        # update last_flops to prevent the search for a new configuration at each iteration
        # search for a new configuration only if the number of flops changes
        self.last_flops = self.flops

        # get the list of configuration names and corresponding flops limits
        key_list = list(self.fpga_models.keys())
        value_list = list(self.fpga_models.values())

        # initially, no better configuration was found
        new_config = False

        # if the number of flops is too low compared to the current range
        # iterate the lists in reverse to obtain the smaller configuration
        # that can contain the neural network model
        if self.flops_w < self.current_range[0]:
            key_list.reverse()
            value_list.reverse()
               
        # obtain the index of the current limit, given the name of the configuration        
        index = key_list.index(self.current_config)

        # if the number of flops exceeds the configuration limit
        # each successive pair of configuration limits represents a possible new range 
        if self.flops_w > self.current_range[1]:
            # starts from the index of the current configuration
            value_list = value_list[index:]

            # iterate on each configuration, starting with the one that could bring an improvement
            for name in key_list[index+1:]:

                # define a new range, with the lower as the current limit and
                # the upper as the limit of the next configuration
                # this's possible, because the configurations are ordered based on limit's values
                lower_limit = value_list[0]['limit']
                current_limit = self.fpga_models[name]['limit']

                # calculates the number of MFLOPS/WATT of the current configuration in the loop
                temp_flops_w = (self.flops / 10**6) / self.fpga_models[name]['power_c']

                # find a configuration that can hold the model
                if (temp_flops_w > lower_limit):
                    # a new configuration has been found
                    # save MFLOPS, name and new range on which operate
                    new_config = True
                    self.flops_w = temp_flops_w
                    self.current_config = name
                    self.current_range = [lower_limit, current_limit]
                    self.power = self.fpga_models[name]['power_c']
                   
                # scrolls upwards to search a new configuration
                value_list = value_list[1:]
        else:
            # reverse list, start from the next configuration that can lead me to an optimisation
            # in particular, a smaller configuration that can hold the model
            value_list = value_list[index+1:]

            # iterate on each configuration, starting with the one that could bring an improvement
            for name in key_list[index+1:]:

                # use the current limit as the upper value of the range
                current_limit = self.fpga_models[name]['limit']

                # move upwards to obtain the next configuration,
                # which it will use to define the lower limit
                # if no more configurations are available, the lower limit is 0
                value_list = value_list[1:]
                if value_list:
                    lower_limit = value_list[0]['limit']
                else: lower_limit = 0

                # calculates the number of MFLOPS/WATT of the current configuration in the loop
                temp_flops_w = (self.flops / 10**6) / self.fpga_models[name]['power_c']

                # find a configuration that can hold the model
                if (temp_flops_w < current_limit):
                    # a new configuration has been found
                    # save MFLOPS, name and new range on which operate
                    new_config = True
                    self.flops_w = temp_flops_w
                    self.current_config = name
                    self.current_range = [lower_limit, current_limit]
                    self.power = self.fpga_models[name]['power_c']

        if new_config:
            print(f" New configuration found: name : {self.current_config}, [MFLOPS/W] : {self.current_range[1]}\n")

    def obtain_values(self):
        # has to match the list of facts
        return {'low_p' : self.current_range[0], 'high_p' : self.current_range[1], 'flops_w' : self.flops_w, 'last_flops' : self.last_flops, 'new_latency' : self.latency}

    def printing_values(self):
        print(colors.FAIL, f"ENERGY: {self.power}, MODEL: {self.flops_w} [MFLOPS/W]", colors.ENDC)
        print(colors.FAIL, f"CONFIG: {self.current_config} : {self.current_range[1]} [MFLOPS/W]", colors.ENDC)
        print(colors.FAIL, f"LATENCY: {self.latency}", colors.ENDC)

    def optimiziation_function(self, *args):
        return -self.power

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

    def get_model_latency(self, model):
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
        nvdla = "nv_large2048_fp32.yaml"
        log_file = "profiler_logs.txt"
        
        # build the path where the configuration file is located
        work_p = os.getcwd()
        config_p = Path(work_p).joinpath('specs').joinpath(nvdla)
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
                
        #print(f"total latency: {total_latency}")

        return total_latency
