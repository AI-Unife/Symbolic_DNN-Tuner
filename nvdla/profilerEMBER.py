import os
import sys
import torch
import yaml
import glob  # --- MODIFICA: Aggiunto import ---
import sys

# Imposta le variabili d'ambiente
os.environ['TORCH_DATASETPATH'] = '/hpc/home/bzzlca/datasets'
os.environ['TORCH_TRAINPATH'] = '/hpc/home/bzzlca/models'

# Aggiungi il tuo percorso 'build' al sys.path
nvdla_build_path = '/hpc/home/bzzlca/NVDLA-EMBER/build'

# Aggiungilo solo se non è già presente (buona pratica)
if nvdla_build_path not in sys.path:
    sys.path.append(nvdla_build_path)



# sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))


from nvdla.models.zoo import models_factory
from nvdla.wrapper.parser.torch import nvdlaReplacer as converter

##################################################################



def profile_network(model, X, config, outdir):

    # Generate Converter
    print(f'[INFO] Enabling Profiler...')
    nvconverter = converter(config, True, outdir)
    nvmodel = nvconverter.replaceAll(model)
    print(nvmodel)

    ## Profile Input
    try:
        with torch.no_grad():
            for x, _ in X:
                X = x
                _ = nvmodel(X)
                break
    except Exception as e:
        _ = nvmodel(X)

    # Layers log
    exectime = 0

    # --- INIZIO MODIFICA: Implementazione della ricerca file come da richiesta ---
    print(f'[INFO] Searching for layer logs in {outdir}...')
    workdir = outdir
    # Trova tutti i file .yaml nella directory di output
    search_pattern = os.path.join(workdir, "*.yaml")
    all_yaml_files = glob.glob(search_pattern)
    
    # Definisci i nomi dei file da escludere
    excluded_filenames = {
        "netcontent.yaml", 
        "layerlog.yaml", 
        "seulog.yaml", 
        "work_specs.yaml"
    }
    
    layerlogs = []
    for f_path in all_yaml_files:
        # Aggiungi alla lista solo se è un file e il suo nome non è nell'elenco degli esclusi
        if os.path.isfile(f_path) and os.path.basename(f_path) not in excluded_filenames:
            layerlogs.append(f_path)
    
    if not layerlogs:
        print(f'[WARNING] No layer log .yaml files found in {outdir}. Total time will be 0.')
    # --- FINE MODIFICA ---

    for layerlog in layerlogs: # Ora 'layerlogs' è definita
        
        with open(layerlog, 'r') as f:
            layer = yaml.load(f, Loader=yaml.SafeLoader)
        
            # Get Time
            if 'total-layertime' in layer:
                exectime += layer['total-layertime']
            else:
                print(f'[WARNING] Key "total-layertime" not found in {layerlog}.')

    # --- INIZIO MODIFICA: Cancellazione dei file .yaml processati ---
    # print(f'[INFO] Cleaning up {len(layerlogs)} processed layer log files...')
    # for layerlog_file in layerlogs:
    #     try:
    #         os.remove(layerlog_file)
    #     except OSError as e:
    #         print(f'[ERROR] Could not delete file {layerlog_file}: {e}')
    # --- FINE MODIFICA ---

    print(f'Total Inference Time: {exectime} cycles of clock')
    if "int8" in config or "fp8" in config:
        # Periodo = 1 ns -> Calcolo = time * 1 * 1e-6
        time_in_ms = exectime * 1 * 1e-6
        print(f'Data type: 8-bit (Period=1ns). Approximate Time: {time_in_ms:.6f} ms')
    elif "int16" in config or "fp16" in config:
        # Periodo = 2 ns -> Calcolo = time * 2 * 1e-6
        time_in_ms = exectime * 2 * 1e-6
        print(f'Data type: 16-bit (Period=2ns). Approximate Time: {time_in_ms:.6f} ms')  
    elif "int32" in config or "fp32" in config:
        # Periodo = 4 ns -> Calcolo = time * 4 * 1e-6
        time_in_ms = exectime * 4 * 1e-6
        print(f'Data type: 32-bit (Period=4ns). Approximate Time: {time_in_ms:.6f} ms')
    else:
        print(f'[WARNING] Data type (int8/16/32) not detected in config name.')
        print(f'           Unable to determine correct clock period (1, 2, or 4 ns).')
        return None
        
    return time_in_ms

if __name__ == '__main__':
    # Imposta le variabili d'ambiente
    os.environ['TORCH_DATASETPATH'] = '/hpc/home/bzzlca/datasets'
    os.environ['TORCH_TRAINPATH'] = '/hpc/home/bzzlca/models'

    # Aggiungi il tuo percorso 'build' al sys.path
    nvdla_build_path = '/hpc/home/bzzlca/NVDLA-EMBER/build'

    # Aggiungilo solo se non è già presente (buona pratica)
    if nvdla_build_path not in sys.path:
        sys.path.append(nvdla_build_path)
    model, testloader = models_factory('mnist', 'lenet', 1, False)
    config = '/hpc/home/bzzlca/NVDLA-EMBER/specs/nv_large1024_int32.yaml'
    time = profile_network(model, testloader, config, './debug/')
    