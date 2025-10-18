from components.model_interface import LayerTypes, Params
from modules.common_interface import common_interface
from components.colors import colors
import os

from pathlib import Path
from tensorflow.keras import layers, models

import nvdla.profiler as profiler
from math import sqrt

class hardware_module(common_interface):
    facts = ['hw_latency', 'max_latency']
    problems = ['out_range']
    weight = 0.5

    def __init__(self, backend, input_shape, n_classes):
        self.backend = backend
        self.input_shape = input_shape
        self.cost_par = 10000
        self.weight_cost = 0.9
        self.max_latency = 0.010  # 10ms

        nvdla_list = [{'name': "nv_small", 'path': "nv_small64_fp32.yaml", 'area': 2.824},
                      {'name': "nv_small256", 'path': "nv_small256_fp32.yaml", 'area': 3.091},
                      {'name': "nv_large", 'path': "nv_large2048_fp32.yaml", 'area': 3.809}]
        self.nvdla = {}

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
        
        self.last_flops = 0
        self.nvdla  = dict(sorted(self.nvdla.items(), key=lambda item: item[1]['cost'], reverse=True))
        first_el = next(iter(self.nvdla))
        self.max_cost = self.nvdla[first_el]['cost']

    def LENET(self):
        return self.backend.build_lenet()

    def update_state(self, *args):
        self.model = args[2]
        self.flops = self.backend.get_flops(self.model, self.input_shape)

        if self.last_flops == self.flops:
            return

        for config_key in self.nvdla:
            config_path = self.nvdla[config_key]['path']
            self.nvdla[config_key]['latency'] = self.get_model_latency(self.model, config_path) / (10**9)
            latency_temp = self.nvdla[config_key]['latency'] / self.max_latency
            cost_temp = self.nvdla[config_key]['cost'] / self.max_cost
            self.nvdla[config_key]['total_cost'] = round((cost_temp * self.weight_cost) + (latency_temp * (1-self.weight_cost)), 4)

        sorted_config = dict(sorted(self.nvdla.items(), key=lambda item: item[1]['total_cost']))
        self.nvdla = sorted_config
        first_el = next(iter(self.nvdla))
        self.latency = self.nvdla[first_el]['latency']
        self.cost = self.nvdla[first_el]['cost']
        self.total_cost = self.nvdla[first_el]['total_cost']
        self.current_config = first_el

        self.last_flops = self.flops

    def get_model_latency(self, model, config_path):
        total_latency = 0
        batch = 1

        work_p = os.getcwd()
        config_p = Path(work_p).joinpath('nvdla').joinpath('specs').joinpath(config_path)
        nvdla_profiler = profiler.nvdla(config_p)
        log_file = "profiler_logs.txt"

        #input_layer_shape = list(self.model.layers.values())[0].output_shape

        #input_size = [batch, input_layer_shape[3], input_layer_shape[1], input_layer_shape[2]]

        for layer_spec in self.model.layers.values():
            if layer_spec.type == LayerTypes.Conv2D:
                out_size = [batch, layer_spec.get(Params.OUT_CHANNELS), layer_spec.get(Params.OUT_HEIGHT), layer_spec.get(Params.OUT_WIDTH)]

                if layer_spec.get(Params.PADDING) == 'valid': 
                    padding = 0
                else:
                    padding = int(layer_spec.get(Params.KERNEL_SIZE)[0] - 1) / 2

                input_size = [batch, layer_spec.get(Params.IN_CHANNELS), layer_spec.get(Params.IN_HEIGHT), layer_spec.get(Params.IN_WIDTH)]

                conv_obj = profiler.Conv2d(nvdla_profiler, log_file, layer_spec.name, out_size, input_size[1],
                                           out_size[1], layer_spec.get(Params.KERNEL_SIZE)[0], layer_spec.get(Params.STRIDE)[0],
                                           padding, 1, layer_spec.get(Params.BIAS))
                total_latency += conv_obj.forward(input_size)

            elif layer_spec.type == LayerTypes.Dense:
                out_size = [batch, layer_spec.get(Params.OUT_FEATURES)]
                dense_obj = profiler.Linear(nvdla_profiler, log_file, layer_spec.name, out_size, layer_spec.get(Params.IN_FEATURES), layer_spec.get(Params.OUT_FEATURES), layer_spec.get(Params.BIAS))
                total_latency += dense_obj.forward([batch, layer_spec.get(Params.IN_FEATURES)])

        return total_latency

    def obtain_values(self):
        # has to match the list of facts
        return {'hw_latency' : self.latency, 'max_latency' : self.max_latency}

    def printing_values(self):
        print(f"LATENCY: {self.latency} s",)
        print(f"CURRENT HW: {self.current_config} [{self.cost}$]")
        print(f"TOTAL COST: {self.total_cost}")

    def optimiziation_function(self, *args):
        return -self.total_cost

    def plotting_function(self):
        pass

    def log_function(self):
        pass