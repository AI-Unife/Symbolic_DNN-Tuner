import os
import sys
import yaml
import tempfile
from pathlib import Path

# path alla cartella TUNER_ROOT
TUNER_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(TUNER_ROOT))

# path alla cartella NVDLA-EMBER
EMBER_PATH = TUNER_ROOT.parent / "NVDLA-EMBER"
sys.path.append(str(EMBER_PATH))

# path alla cartella dei file di configurazione dei backend hardware
SPECS_PATH = TUNER_ROOT / "nvdla" / "specs"
EMBER_SPECS_PATH = EMBER_PATH / "NVDLA-EMBER" / "specs" / "hw_configs"
TEMP_LOG_DIR = EMBER_PATH / "test_logs"


from components.colors import colors
from modules.loss.generate_minimal_cfg import generate_minimal_cfg
from modules.common_interface import common_interface
from components.model_interface import LayerTypes, Params
from pytorch_implementation.module_backend import isConv2d, get_dummy_input, extract_conv_layers

import nvdla.profiler as profiler
from exp_config import load_cfg
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
        
        elif hw_backend == "ember":

            # directory with yaml files
            self.specs_dir = Path(EMBER_SPECS_PATH)

            # init list of available configurations
            self.nvdla = {}

            # search for yaml files in the directory
            yaml_files = list(self.specs_dir.glob("*.yaml"))

            if not yaml_files:
                raise ModuleNotFoundError("No YAML configuration files found in EMBER directory")

            for spec_file in yaml_files:

                try:
                    with open(spec_file, "r") as f:
                        spec = yaml.safe_load(f)
                except Exception as e:
                    print(colors.FAIL, f"Error reading {spec_file.name}: {e}", colors.ENDC)
                    continue
                
                config_name = spec.get("name", spec_file.stem)

                current_cost = 0

                # parameter extraction
                max_batch = spec.get("cmac", {}).get("batch-size", 1)

                # Feature buffer entries
                datbuf = spec.get("cbuf", {}).get("datbuf", {})
                ft_buf_entries = (
                    datbuf.get("num-banks", 0) *
                    datbuf.get("bank-depth", 0)
                )

                # Output buffer entries
                cacc = spec.get("cacc", {})
                out_buf_entries = (
                    cacc.get("num-banks", 0) *
                    cacc.get("bank-depth", 0)
                )

                # saving the configuration info in the nvdla dict
                self.nvdla[config_name] = {
                    'path': spec_file.name,
                    'cost': 0,
                    'latency': 0,
                    'total_cost': 0,
                    'dtype': spec.get("dat-type", "unknown"),

                    # hw metrics for optimization suggestions
                    'max_batch': max_batch,
                    'ft_buf_entries': ft_buf_entries,
                    'out_buf_entries': out_buf_entries
                }

        if not self.nvdla:
            raise ModuleNotFoundError("No valid NVDLA configuration found")

        if self.nvdla == {}:
            raise ModuleNotFoundError("No NVDLA configuration found")

        # maximum cost based on the latency
        self.nvdla  = dict(sorted(self.nvdla.items(), key=lambda item: item[1]['latency'], reverse=True))

        # flag to use or not hw cost
        self.use_hw_cost = self.cfg.get('use_hw_cost', True)

        #flag to accept or not suggestion for hw optimization
        self.suggest_hw_opt = self.cfg.get('suggest_hw_opt', True)

    
    def update_state(self, model, input_shape=None):
        # import current model reference
        self.model = model
        self.input_shape = input_shape
        list_to_pop = []

        # for each configuration calculate the latency and the total cost
        for config_key in self.nvdla:
            
            config_path = self.specs_dir / self.nvdla[config_key]['path']
            d_type = self.nvdla[config_key]['dtype']

            # skip the configuration if it doesen't respect some constraints
            if (self.cfg.hw_backend == "ember") and (not self.hw_supports_net(self.model, self.nvdla[config_key]) or d_type != "int8"): 
                #remove the configuration from the list of available configurations
                list_to_pop.append(config_key)
                print(colors.WARNING, f"\n[WARN] {config_key} CONFIGURATION NOT COMPATIBLE WITH THE CURRENT MODEL\n", colors.ENDC)
                continue
            else:
                print(colors.OKGREEN, f"\n[OK] {config_key} CONFIGURATION COMPATIBLE WITH THE CURRENT MODEL\n", colors.ENDC)

            self.nvdla[config_key]['latency'] = self.get_model_latency(self.model, config_path) / (10**9)
            
            # use hw latency only if the flag is true 
            latency_temp = self.nvdla[config_key]['latency'] / self.max_latency
            if self.use_hw_cost:
                cost_temp = self.nvdla[config_key]['cost'] / self.max_cost
                total = (cost_temp * self.weight_cost) + (latency_temp * (1 - self.weight_cost))
            else:
                total = latency_temp  # solo latenza

            self.nvdla[config_key]['total_cost'] = round(total, 4)

        # remove the configurations that are not compatible with the current model
        for config_key_to_pop in list_to_pop:
            self.nvdla.pop(config_key_to_pop)

        # sort the configurations by cost
        print(colors.OKBLUE, f"\n[INFO] Available hardware configurations after compatibility check:", colors.ENDC)
        for config_key in self.nvdla:
            print(f"  - {config_key}: latency={self.nvdla[config_key]['latency']:.6f} s, cost={self.nvdla[config_key]['cost']}$, total_cost={self.nvdla[config_key]['total_cost']}")
        
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
        print(f"\nLATENCY: {self.latency} s",)
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

        #flag Pytorch o TF
        framework = self.cfg.get("backend", "torch")
        if framework not in ["torch", "tf"]:
            raise ValueError(f"Framework non supportato: {framework}")
        
        hw_backend = self.cfg.hw_backend

        if hw_backend == "nvdla":
            work_p = os.getcwd()
            config_p = Path(work_p).joinpath('nvdla').joinpath('specs').joinpath(config_path)
            nvdla_profiler = profiler.nvdla(config_p)
            log_file = "profiler_logs.txt"

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

            #model = SimpleCNN() 
            dummy_input = get_dummy_input(self) 
        
            # esecuzione del profiler_ember
            total_latency = ember_profile_network(
                model,
                dummy_input,
                str(config_path),
                str(TEMP_LOG_DIR)
            )
                
            return total_latency

        if hw_backend == "ember" and self.cfg.backend != "torch":
            raise RuntimeError("EMBER requires backend=torch")
        
        if hw_backend == "nvdla" and self.cfg.beckend != "tensorflow":
            raise RuntimeError("NVDLA requires backend=tf")

        else:
            raise ValueError(f"Unsupported hw_backend: {hw_backend}")


    def suggest_optimization(self, model):
        """
        Suggest hardware optimization by creating a minimal configuration
        file that meets the model requirements."""

        if self.suggest_hw_opt:

            print("\n[INFO] Suggesting hardware optimization...")
            print(f"Current configuration: {self.current_config} with latency {self.latency} s and cost {self.cost}$")

            generate_minimal_cfg(self, model)


    def hw_supports_net(self, model, dict_config):
        """
        Checks if the hardware specified by dict_config can run the model.
        Uses forward hooks to extract the tensor dimensions of Conv2D layers.
        Batch size is set to 1."""

        max_batch = dict_config.get("max_batch", 1)
        ft_buf_entries = dict_config.get("ft_buf_entries", 0)
        out_buf_entries = dict_config.get("out_buf_entries", 0)
        compatible = True

        first_conv = next((m for m in model.modules() if isConv2d(m)), None)
        if first_conv is None:
            print(colors.WARNING, f"\n[WARN] No Conv2d layer found in model", colors.ENDC)
            return False

        conv_layers = extract_conv_layers(self, model)

        for idx, (input_shape, output_shape) in enumerate(conv_layers):
            B, C, H, W = input_shape
            entries_feat = H * W * min(B, max_batch)
            if entries_feat > ft_buf_entries:
                print(colors.WARNING, f"\n[WARN] Layer {idx} -> Feature buffer overflow: {entries_feat} > {ft_buf_entries}", colors.ENDC)
                compatible = False

            B, K, H_out, W_out = output_shape
            entries_out = H_out * W_out * min(B, max_batch)
            if entries_out > out_buf_entries:
                print(colors.WARNING, f"\n[WARN] Layer {idx} -> Output buffer overflow: {entries_out} > {out_buf_entries}", colors.ENDC)
                compatible = False

        return compatible

if __name__ == "__main__":
    hw_module = hardware_module()
    model = SimpleCNN()
    hw_module.update_state(model, input_shape=(3, 32, 32))
    hw_module.printing_values()
    hw_module.suggest_optimization(model)