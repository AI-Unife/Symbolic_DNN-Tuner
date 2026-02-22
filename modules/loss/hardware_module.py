import os
import sys
import torch
import tempfile
import torch.nn as nn
from pathlib import Path
from components.colors import colors
from modules.common_interface import common_interface
from components.model_interface import LayerTypes, Params, LayerSpec

import nvdla.profiler as profiler
from exp_config import load_cfg

# path alla cartella NVDLA-EMBER
PROJECT_ROOT = Path(__file__).resolve().parents[2]
EMBER_PATH = PROJECT_ROOT.parent / "NVDLA-EMBER"
SPECS_PATH = PROJECT_ROOT / "nvdla" / "specs"
EMBER_SPECS_PATH = EMBER_PATH / "NVDLA-EMBER" / "specs"

if EMBER_PATH.exists():
    sys.path.append(str(EMBER_PATH))
else:
    raise FileNotFoundError(f"NVDLA-EMBER not found at: {EMBER_PATH}")

from profiler_ember import SimpleCNN, profile_network as ember_profile_network

class hardware_module(common_interface):

    #facts and problems for creating the prolog model
    facts = ['hw_latency', 'max_latency']
    problems = ['out_range']
    
    #weight of the module for the final loss calculation
    try:
        cfg = load_cfg()
        # weight of the module for the final loss calculation
        weight = cfg.get('w_HW', 0.33)
    except:
        weight = 0.33

    def __init__(self, weight_cost: float = 0.3):    
        # cost value per square millimeter, 10K / mm2
        self.cost_par = 10000
        # attribute indicating how much cost weighs against latency value
        self.weight_cost = weight_cost
        # max latency value in second 
        self.max_latency = 0.008 #120FPS,
        # max manifacturing cost value
        self.max_cost = 30000
        try:
            self.cfg = load_cfg()
        except:
            self.cfg = {"name": "./"}

        hw_backend = self.cfg.hw_backend
        if hw_backend == "nvdla":
            nvdla_list = [{'name': "nv_small", 'path': "nv_small64_fp32.yaml", 'area': 2.824},
                          {'name': "nv_small256", 'path': "nv_small256_fp32.yaml", 'area': 3.091},
                          {'name': "nv_large", 'path': "nv_large2048_fp32.yaml", 'area': 3.809}]
            
            self.specs_dir = SPECS_PATH
        
        elif hw_backend == "ember":
            nvdla_list = [
                {'name': "nv_small64",  'path': "nv_small64_int8.yaml",    'area': 2.824},
                {'name': "nv_small256", 'path': "nv_small256_int8.yaml",   'area': 3.091},
                {'name': "nv_large2048",'path': "nv_large2048_int8.yaml",  'area': 3.809},
            ]   

            self.specs_dir = EMBER_SPECS_PATH

        # init list of available configurations to an empty dict
        self.nvdla = {}
        # iterate over each configuration
        for config in nvdla_list:
            spec_file = self.specs_dir / config['path']
            if spec_file.exists():
                # calculate the current manifacturing cost
                current_cost = round(self.cost_par*config['area'], 2)
                # inclusion of only configurations that are less expensive than the cost limit
                #if current_cost <= self.max_cost:
                self.nvdla[config['name']] = {'path': config['path'],
                                                'cost': current_cost,
                                                'latency': 0,
                                                'total_cost': 0}
            else:
                print(colors.FAIL, f"|  --------- {config['name']} CONFIGURATION FILE DOESN'T EXIST  -------  |\n", colors.ENDC)
        
        if self.nvdla == {}:
            raise ModuleNotFoundError("No NVDLA configuration found")

        # maximum cost
        self.nvdla  = dict(sorted(self.nvdla.items(), key=lambda item: item[1]['cost'], reverse=True))

        # flag to use or not hw cost
        self.use_hw_cost = self.cfg.get('use_hw_cost', True)

    
    def update_state(self, *args):
        # import current model reference
        self.model = args[0]

        # for each configuration calculate the latency and the total cost
        for config_key in self.nvdla:
            
            config_path = self.specs_dir / self.nvdla[config_key]['path']
            self.nvdla[config_key]['latency'] = self.get_model_latency(self.model, config_path) / (10**9)
            
            # use hw latency only if the flag is true 
            latency_temp = self.nvdla[config_key]['latency'] / self.max_latency
            if self.use_hw_cost:
                cost_temp = self.nvdla[config_key]['cost'] / self.max_cost
                total = (cost_temp * self.weight_cost) + (latency_temp * (1 - self.weight_cost))
            else:
                total = latency_temp  # solo latenza

            self.nvdla[config_key]['total_cost'] = round(total, 4)
        
        # sort the configurations by cost
        # this will be useful to determine the optimal configuration
        sorted_config = dict(sorted(self.nvdla.items(), key=lambda item: item[1]['total_cost']))
        self.nvdla = sorted_config
        first_el = next(iter(self.nvdla))
        self.latency = self.nvdla[first_el]['latency']
        self.cost = self.nvdla[first_el]['cost']
        self.total_cost = self.nvdla[first_el]['total_cost']
        self.current_config = first_el


    def obtain_values(self):
        # has to match the list of facts
        return {'hw_latency' : self.latency, 'max_latency' : self.max_latency}


    def printing_values(self):
        print(f"LATENCY: {self.latency} s",)
        print(f"CURRENT HW: {self.current_config} [{self.cost}$]")
        print(f"TOTAL COST: {self.total_cost}")


    def optimiziation_function(self, *args):
        return self.total_cost

    def plotting_function(self):
        pass

    def log_function(self):
        f = open("{}/algorithm_logs/hardware_report.txt".format(self.cfg.name), "a")
        f.write(str(self.latency) + "," + str(self.cost) + "," + str(self.total_cost) + "," + str(
            self.current_config) + "\n")
        f.close()


    def get_model_latency(self, model, config_path):
        total_latency = 0
        batch = 1

        work_p = os.getcwd()
        config_p = Path(work_p).joinpath('nvdla').joinpath('specs').joinpath(config_path)
        nvdla_profiler = profiler.nvdla(config_p)
        log_file = "profiler_logs.txt"

        #flag Pytorch o TF
        framework = self.cfg.get("backend", "torch")
        if framework not in ["torch", "tf"]:
            raise ValueError(f"Framework non supportato: {framework}")
        
        hw_backend = self.cfg.hw_backend

        if hw_backend == "nvdla":
            print("[INFO] Using DEFAULT hardware backend")
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
        
        elif hw_backend == "ember":
            print("[INFO] Using EMBER hardware backend")

            #set the path to the configuration file of the hardware backend
            ember_config = EMBER_SPECS_PATH / "nv_large2048_int8.yaml"

            #model = SimpleCNN() 
            dummy_input = torch.randint(0, 256, (1, 3, 32, 32)) 
           
            # directory temporanea per i log
            with tempfile.TemporaryDirectory() as outdir:
                # esecuzione del profiler_ember
                total_latency = ember_profile_network(
                    model,
                    dummy_input,
                    str(ember_config),
                    outdir
                )
                
            return total_latency

        if hw_backend == "ember" and self.cfg.backend != "torch":
            raise RuntimeError("EMBER requires backend=torch")
        
        if hw_backend == "nvdla" and self.cfg.beckend != "tensorflow":
            raise RuntimeError("NVDLA requires backend=tf")

        else:
            raise ValueError(f"Unsupported hw_backend: {hw_backend}")

if __name__ == "__main__":
    hw_module = hardware_module()
    model = SimpleCNN() 
    hw_module.update_state(model)
    hw_module.printing_values()

