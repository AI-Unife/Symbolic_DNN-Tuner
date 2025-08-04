import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from modules.loss.hardware_module import hardware_module
import nvdla.profiler as profiler
from components.colors import colors



class custom_hardware_module(hardware_module):
    def __init__(self):
        # custom hardware module initialization
        super().__init__()

        specs_dir= 'nvdla/specs'
        nvdla_list = []
        
        # iterate over each file in the specs directory
        for filename in os.listdir(specs_dir):
            if filename.endswith('.yaml'):
                full_path = os.path.join(specs_dir, filename)
                config_obj = profiler.nvdla(full_path)
                name = config_obj.config['name']
                area = config_obj.config.get('area', 1.0)  # <-- ???
                nvdla_list.append({'name': name, 'path': filename, 'area': area})
        
        # init list of available configurations to an empty dict
        self.nvdla = {}

        for config in nvdla_list:
            if os.path.exists('nvdla/specs/' + config['path']):
                # calculate the current manifacturing cost
                current_cost = round(self.cost_par * config['area'], 2)
                # inclusion of only configurations that are less expensive than the cost limit
                if current_cost <= self.max_cost:
                    self.nvdla[config['name']] = {'path': config['path'],
                                                  'cost': current_cost,
                                                  'latency': 0,
                                                  'total_cost': 0}
            else:
                print(colors.FAIL, f"|  --------- {config['name']} CONFIGURATION FILE DOESN'T EXIST  -------  |\n", colors.ENDC)        

        if self.nvdla == {}:
            raise ModuleNotFoundError("No NVDLA configuration found")