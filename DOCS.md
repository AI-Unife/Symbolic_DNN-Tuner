# Astrazione del tuner e sviluppo di nuovi backend

Questo branch contiene l'astrazione del tuner e due implementazioni di esempio:

- `tensorflow_implementation/`: implementazione su TensorFlow/Keras
- `pytorch_implementation/`: implementazione di esempio su PyTorch

L'obiettivo dell'astrazione è separare:
1) Logica del tuner (ottimizzazione bayesiana, diagnosi simbolica, regole di tuning, ecc...)  
2) Dettagli specifici del framework (creazione modello, training loop, salvataggio checkpoint, calcolo FLOPs, ecc...)

## Struttura del progetto

- `components/`: core del tuner (agnostico rispetto al framework)
  - `tuner.py`: orchestratore principale (BO e diagnosi)
  - `controller.py`: coordina training, moduli di loss e parte simbolica
  - `neural_network.py`: classe astratta per training e operazioni di modifica architettura
  - `model_interface.py`: rappresentazione astratta del modello (`TunerModel`, `LayerSpec`, `LayerTypes`, ecc...)
  - `backend_interface.py`: interfaccia backend per i moduli (`BackendInterface`)
- `modules/`: "loss modules" (es. `hardware_module`, `accuracy_module`) che consumano modello/backend
- `tensorflow_implementation/`: backend TensorFlow (modello, training, FLOPs, launcher)
- `pytorch_implementation/`: backend PyTorch (modello, training, FLOPs, launcher)
- `main.py`: entrypoint "vuoto" che rimanda ai launcher framework-specifici

## Cosa è stato reso astratto

### 1) Orchestrazione del tuner tramite dependency injection (`TunerConfig`)

Il tuner non istanzia più direttamente classi TensorFlow, ma riceve una configurazione:

- `components/tuner.py`: `TunerConfig` contiene:
  - `neural_network_cls`: implementazione framework-specifica del training
  - `module_backend_cls`: backend per i moduli (FLOPs / info layer / ecc...)
  - `dataset`: dataset già caricato (wrapper comune)
  - configurazioni opzionali: `search_space`, `fixed_hyperparams`, `clear_session_callback`, ecc...

Il `Tuner` usa questi oggetti e resta invariato rispetto alla logica (BO e diagnosi).

### 2) Interfaccia astratta del modello (`TunerModel`) e layer "standardizzati"

Per poter modificare l'architettura (aggiungere/rimuovere sezioni Conv/Dense/BatchNorm, ecc...) in modo indipendente dal framework:

- `components/model_interface.py` definisce:
  - `LayerTypes`: elenco di tipi "standard" (ispirato ai layer Keras)
  - `LayerSpec`: descrizione di un layer (tipo, nome, parametri, flag `is_activation`)
  - `Params`: chiavi standard per i parametri delle `LayerSpec` (es. `in_channels`, `kernel_size`, …)
  - `TunerModel`: classe astratta che richiede metodi per:
    - costruire/aggiornare `layers: Dict[str, LayerSpec]`
    - inserire layer (`add_layers`)
    - rimuovere layer (`remove_layers`)
    - convertire da/verso tipi specifici dei framework (`from_type`, `to_type`, `from_spec`, `to_spec`)

In pratica: ogni backend implementa un wrapper/modello che "parla" il linguaggio del tuner tramite `LayerSpec`.

### 3) Training astratto (`components/neural_network.NeuralNetwork`)

`components/neural_network.NeuralNetwork` è una classe astratta che contiene:
- logica comune di:
  - caricamento dell'ultimo checkpoint compatibile (`checkpoint_format` e `Model/manifest-*.json`)
  - inserimento/rimozione di sezioni Conv/Dense (basato su `LayerTypes` e `DynamicNet`)
- metodi astratti da implementare per ogni backend:
  - `from_checkpoint(manifest)`: caricamento del modello salvato su disco
  - `from_scratch(input_shape, n_classes, params)`: creazione di una nuova istanza del modello
  - `insert_batch(model, params)`: inserire batch normalizazion/regolarizzazione
  - `training(...)`: training loop vero e proprio

### 4) Backend per i moduli (`BackendInterface`)

Alcuni moduli (es. `modules/loss/hardware_module.py`) hanno bisogno di funzioni "di servizio" che dipendono dal framework (es. calcolo FLOPs).

- `components/backend_interface.py` definisce `BackendInterface`
- ogni implementazione fornisce `get_flops(...)` e (se utile) mapping/inspection dei layer

## Come si usa (TensorFlow / PyTorch)

Gli entrypoint sono:

- TensorFlow: `python tensorflow_implementation/main.py`
- PyTorch: `python pytorch_implementation/main.py`

Entrambi:

1) creano un `TunerDataset` e caricano un dataset (es. CIFAR-10)
2) costruiscono `TunerConfig` puntando alle classi del backend
3) eseguono `Tuner(config).run()`

Nota: i launcher aggiungono il project root a `sys.path` per permettere import assoluti quando eseguiti come script. Altrimenti i due file si sarebbero dovuti eseguire come moduli.

### Implementazioni presenti

- TensorFlow (`tensorflow_implementation/`)
  - `tensorflow_implementation/main.py`: costruisce `TunerConfig` (include `clear_session_callback=K.clear_session`)
  - `tensorflow_implementation/neural_network.py`: training loop Keras e salvataggio `model.json`/weights e manifest
  - `tensorflow_implementation/model.py`: wrapper `TFModel(TunerModel)` che supporta insert/remove layer via `LayerSpec`
  - `tensorflow_implementation/module_backend.py`: `get_flops()` tramite `tensorflow_implementation/flops/`

- PyTorch (`pytorch_implementation/`)
  - `pytorch_implementation/main.py`: costruisce `TunerConfig` (nell'esempio usa `fixed_hyperparams` per delle esecuzioni riproducibili)
  - `pytorch_implementation/neural_network.py`: training loop PyTorch e salvataggio full model (`model_path`) e manifest
  - `pytorch_implementation/model.py`: wrapper `TorchModel(TunerModel, nn.Module)` con `ModuleList` e ricostruzione layer
  - `pytorch_implementation/module_backend.py`: `get_flops()` via `torch.utils.flop_counter`

## Come implementare un nuovo backend (es. JAX, MindSpore, ecc...)

Prendere come riferimento gli esempi Tensorflow e Python.

### Step 1: creare una sottoclasse di `TunerModel` (`model.py`)

Creare una classe (es. `MyFrameworkModel`) che:
- estende `TunerModel`
- mantiene un dizionario `self.layers: Dict[str, LayerSpec]` aggiornato (`create_specs`)
- implementa:
  - `add_layers(layers: List[LayerSpec], targets: List[LayerTypes], position: InsertPosition)`
  - `remove_layers(target: LayerSpec, linked_layers: List[LayerTypes], delimiter: bool, first_found: bool)`
  - `from_type(layer_type: LayerTypes)` / `to_type(cls_or_layer)` / `from_spec(layer_spec: LayerSpec)`

Requisiti minimi (per le operazioni attualmente usate):

- supporto a `LayerTypes.Conv2D`, `Dense`, `Flatten`, `Dropout`, `MaxPooling2D`, `BatchNormalization*`, `Activation`
- `LayerSpec.params` deve includere i campi usati da `hardware_module`:
  - per Conv2D: `in_channels`, `out_channels`, `in_height`, `in_width`, `out_height`, `out_width`, `kernel_size`, `stride`, `padding`, `bias`
  - per Dense: `in_features`, `out_features`, `bias`

Come in `pytorch_implementation/model.py`, spesso serve una funzione "shape-fixer" (es. `fix_shapes`) per adattare automaticamente `in_channels` / `in_features` quando si inseriscono nuovi layer.

### Step 2: implementare il training (`neural_network.py`)

Crea una classe che estende `components.neural_network.NeuralNetwork` e:

- imposti `self.checkpoint_format` (stringa unica, es. `"jax"` o `"mindspore"`)
- definisca:
  - preprocessing dataset (es. channel-last vs channel-first, one-hot, dtype)
  - mapping di attivazioni (`self.activation_map`) e optimizer (`self.optimizer_map`) coerenti col search space
- implementi:
  - `from_scratch(...)`: costruisce e restituisce un `TunerModel`
  - `from_checkpoint(manifest)`: ripristina il modello dall'ultimo `manifest-*` compatibile
  - `insert_batch(...)`: aggiunge BatchNorm/regolarizzazione
  - `training(...)`: esegue training e validation. Ritorna:
    - `score`: `[best_val_loss, best_val_acc]` (come negli esempi)
    - `history`: dizionario con chiavi `loss`, `val_loss`, `accuracy`, `val_accuracy`
    - `model`: istanza del tuo `TunerModel` (non solo il "raw model" del framework)

Checkpoint/manifest:
- salva i pesi/modello in `Model/` e/o `Weights/`
- chiama `self.save_manifest({...})` (definito in `components/neural_network.py`) includendo:
  - `params`, `input_shape`, `n_classes`
  - puntatori al file pesi/modello (framework-specifici, es. `model_path` nel backend PyTorch)

### Step 3: implementare il backend per i moduli (`module_backend.py`)

Implementa `components.backend_interface.BackendInterface`, in particolare:

- `get_flops(model, input_shapes)`:
  - deve restituire un numero (FLOPs totali) coerente con il tuo framework/modello
  - nota: `hardware_module` passa il `TunerModel` restituito dal training
- opzionali: `get_layer_info`, `get_layers`, `get_input_shape`, `get_output_shape`

### Step 4: creare un launcher (`main.py`)

Il launcher deve:
- creare e popolare `TunerDataset` (o caricare dataset custom con `load_custom_dataset`)
- creare `TunerConfig(neural_network_cls=..., module_backend_cls=..., dataset=..., ...)`
- invocare `Tuner(config).run()`
