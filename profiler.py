import os
import sys
import torch
import yaml
import glob  # --- MODIFICA: Aggiunto import ---
import sys
import tensorflow as tf
import tf2onnx
import onnx
import onnx2torch

# Imposta le variabili d'ambiente
os.environ['TORCH_DATASETPATH'] = '/hpc/home/bzzlca/datasets'
os.environ['TORCH_TRAINPATH'] = '/hpc/home/bzzlca/models'

# Aggiungi il tuo percorso 'build' al sys.path
nvdla_build_path = '/hpc/home/bzzlca/NVDLA-EMBER/build'

# Aggiungilo solo se non è già presente (buona pratica)
if nvdla_build_path not in sys.path:
    sys.path.append(nvdla_build_path)



sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))


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
    if isinstance(X, torch.Tensor):
        _ = nvmodel(X)
    else:
        with torch.no_grad():
            for x, _ in X:
                X = x
                _ = nvmodel(X)
                break

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

class SimpleCNN(torch.nn.Module):
    """
    Una semplice architettura di rete neurale convoluzionale.
    """
    def __init__(self):
        super(SimpleCNN, self).__init__()
        # QuantStub converte l'input da float32 a int8
        self.quant = torch.quantization.QuantStub()
        
        # --- Blocco Convoluzionale 1 ---
        # Input: (Batch Size, 3 canali, 32, 32)
        # Output: (Batch Size, 16 canali, 32, 32)
        self.conv1 = torch.nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1)
        # 'padding=1' con 'kernel_size=3' mantiene le dimensioni 32x32
        
        # --- Blocco di Pooling 1 ---
        # Input: (Batch Size, 16, 32, 32)
        # Output: (Batch Size, 16, 16, 16)
        self.pool = torch.nn.MaxPool2d(kernel_size=2, stride=2)
        
        # --- Blocco Convoluzionale 2 ---
        # Input: (Batch Size, 16, 16, 16)
        # Output: (Batch Size, 32, 16, 16)
        # self.conv2 = torch.nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
        
        # --- Blocco di Pooling 2 ---
        # (Riutilizziamo lo stesso self.pool)
        # Input: (Batch Size, 32, 16, 16)
        # Output: (Batch Size, 32, 8, 8)
        
        # --- Strati Fully Connected (Linear) ---
        # Appiattiamo l'output del blocco pool2
        # La dimensione appiattita è 32 (canali) * 8 (altezza) * 8 (larghezza) = 2048
        # self.fc1 = torch.nn.Linear(32 * 8 * 8, 128)
        
        # Layer di output (10 classi)
        self.fc2 = torch.nn.Linear(16 * 8 * 8, 10)
        self.dequant = torch.quantization.DeQuantStub()

    def forward(self, x):
        """
        Definisce il passaggio "forward" dei dati attraverso la rete.
        """
        
        x = self.quant(x)
        
        # Blocco 1: Conv -> ReLU -> Pool
        x = self.pool(torch.nn.functional.relu(self.conv1(x)))
        
        # Blocco 2: Conv -> ReLU -> Pool
        # x = self.pool(torch.nn.functional.relu(self.conv2(x)))
        
        # Appiattimento dei dati per gli strati lineari
        # 'x.size(0)' è la dimensione del batch (di solito -1 per automatico)
        x = torch.flatten(x, 1) # Appiattisce tutto tranne la dimensione del batch
        
        # Strato lineare 1 con ReLU
        # x = torch.nn.functional.relu(self.fc1(x))
        
        # Strato lineare 2 (output - logits)
        # Non applichiamo ReLU qui perché l'output va a una funzione di perdita
        # come CrossEntropyLoss, che si aspetta i "logits".
        x = self.fc2(x)
        
        x = self.dequant(x)
        
        return x
    

if __name__ == '__main__':
    # Imposta le variabili d'ambiente
    # os.environ['TORCH_DATASETPATH'] = '/hpc/home/bzzlca/datasets'
    # os.environ['TORCH_TRAINPATH'] = '/hpc/home/bzzlca/models'
    is_torch = True
    # Aggiungi il tuo percorso 'build' al sys.path
    nvdla_build_path = '/hpc/home/bzzlca/NVDLA-EMBER/build'

    # Aggiungilo solo se non è già presente
    if nvdla_build_path not in sys.path:
        sys.path.append(nvdla_build_path)
    
    if is_torch:
        model = SimpleCNN()
        dummy_input = torch.randint(0, 256, (1, 3, 32, 32))
    else:
        tf_model = tf.keras.models.load_model('/hpc/home/bzzlca/Symbolic_DNN-Tuner/25_11_12_05_55990_CIFAR-10_filtered_7_1000_50/Model/best-model.keras')
        input_size = tf_model.layers[0].output.shape
        input_size = [1, input_size[3], input_size[1], input_size[2]]
        dummy_input = torch.Tensor(torch.randn(input_size))
        onnx_model = tf2onnx.convert.from_keras(tf_model, output_path="debug/model.onnx")
        onnx_model = onnx.load("debug/model.onnx")
        print("[INFO] ONNX model created.")
        model = onnx2torch.convert(onnx_model)
            
    config = '/hpc/home/bzzlca/NVDLA-EMBER/specs/nv_large2048_int8_modified.yaml'
    time = profile_network(model, dummy_input, config, './debug/')
    