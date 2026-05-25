# Symbolic DNN-Tuner — Documentazione Tecnica


---

## Indice

1. [Panoramica del Progetto](#1-panoramica-del-progetto)
2. [Architettura del Sistema](#2-architettura-del-sistema)
3. [Struttura delle Directory](#3-struttura-delle-directory)
4. [Configurazione (`exp_config.py`)](#4-configurazione-exp_configpy)
5. [Entry Point Principale (`symbolic_tuner.py`)](#5-entry-point-principale-symbolic_tunerpy)
6. [Componenti Core (`components/`)](#6-componenti-core-components)
   - 6.1 [Controller](#61-controller)
   - 6.2 [Search Space](#62-search-space)
   - 6.3 [Objective Function e Constraints](#63-objective-function-e-constraints)
   - 6.4 [Neural Network (Abstract)](#64-neural-network-abstract)
   - 6.5 [Model Interface e Backend Interface](#65-model-interface-e-backend-interface)
   - 6.6 [Neural-Symbolic Bridge](#66-neural-symbolic-bridge)
   - 6.7 [Tuning Rules Symbolic](#67-tuning-rules-symbolic)
   - 6.8 [LFI Integration](#68-lfi-integration)
   - 6.9 [Dataset e Gesture Dataset](#69-dataset-e-gesture-dataset)
   - 6.10 [Improvement Checker](#610-improvement-checker)
   - 6.11 [Integral](#611-integral)
   - 6.12 [Random Search](#612-random-search)
   - 6.13 [Storing Experience](#613-storing-experience)
   - 6.14 [Bayesian Optimization (`bo/`)](#614-bayesian-optimization-bo)
7. [Moduli di Loss (`modules/`)](#7-moduli-di-loss-modules)
   - 7.1 [Module Manager](#71-module-manager)
   - 7.2 [Common Interface](#72-common-interface)
   - 7.3 [FLOPs Module](#73-flops-module)
   - 7.4 [Hardware Module](#74-hardware-module)
8. [Implementazioni Framework-Specific](#8-implementazioni-framework-specific)
   - 8.1 [PyTorch](#81-pytorch)
   - 8.2 [TensorFlow](#82-tensorflow)
9. [Ragionamento Simbolico (`symbolic_base/`)](#9-ragionamento-simbolico-symbolic_base)
10. [FLOPs & Profilazione Hardware](#10-flops--profilazione-hardware)
    - 10.1 [Calculator FLOPs (`flops/`)](#101-calculator-flops-flops)
    - 10.2 [Profiler NVDLA (`nvdla/`)](#102-profiler-nvdla-nvdla)
11. [Quantizzazione (`quantizer/`)](#11-quantizzazione-quantizer)
12. [Script di Utilità](#12-script-di-utilità)
13. [Esecuzione su Cluster HPC (SLURM)](#13-esecuzione-su-cluster-hpc-slurm)
14. [Struttura dei Log di Algoritmo](#14-struttura-dei-log-di-algoritmo)
15. [Interfaccia Grafica (GUI)](#15-interfaccia-grafica-gui)
    - 15.1 [Avvio della GUI](#151-avvio-della-gui)
    - 15.2 [Fasi Operative](#152-fasi-operative-della-gui)
    - 15.3 [Classi Principali](#153-classi-principali)
    - 15.4 [Funzioni di Visualizzazione](#154-funzioni-di-visualizzazione-graphfunpy)
    - 15.5 [Funzioni di Analisi](#155-funzioni-di-analisi-analyze_outpy)
    - 15.6 [Stato Streamlit](#156-stato-streamlit-session-state)
    - 15.7 [Gestione File Dialog](#157-gestione-file-dialog)
    - 15.8 [Modalità di Salvataggio](#158-modalità-di-salvataggio)
    - 15.9 [Dialoghi Informativi](#159-dialoghi-informativi)
16. [Debug Mode](#16-debug-mode)
17. [Dipendenze](#17-dipendenze)
18. [Riferimenti Bibliografici](#18-riferimenti-bibliografici)

---

## 1. Panoramica del Progetto

**Symbolic DNN-Tuner** è un sistema di ottimizzazione automatica degli iperparametri (AutoML) per reti neurali profonde che combina:

- **Bayesian Optimization (BO)** o **Random Search (RS)** per il campionamento degli iperparametri;
- **Ragionamento simbolico** tramite ProbLog per la diagnosi intelligente dei problemi di training e la proposta di azioni correttive;
- **Learning From Interpretations (LFI)** per l'adattamento probabilistico dei pesi delle regole simboliche sulla base dell'esperienza accumulata;
- **Funzioni di loss modulari** per l'ottimizzazione multi-obiettivo (accuracy, FLOPs, latenza hardware).

Il sistema supporta due backend deep learning (TensorFlow e PyTorch), gestisce dataset standard (CIFAR-10/100, MNIST, TinyImageNet) e dataset event-based per gesti (DVSGesture), e include profilazione hardware su NVDLA e quantizzazione opzionale.

---

## 2. Architettura del Sistema

```
┌─────────────────────────────────────────────────────────┐
│                   symbolic_tuner.py                      │
│              (Orchestratore principale)                   │
└────────────────────────┬────────────────────────────────┘
                         │
         ┌───────────────▼────────────────┐
         │     exp_config.py              │
         │  (Configurazione YAML)         │
         └───────────────┬────────────────┘
                         │
         ┌───────────────▼──────────────────────────────┐
         │          LOOP DI OTTIMIZZAZIONE               │
         │     (max_iter iterazioni o convergenza)       │
         └───────────────┬──────────────────────────────┘
                         │
    ┌────────────────────┼────────────────────────┐
    ▼                    ▼                        ▼
┌────────┐     ┌──────────────┐         ┌──────────────┐
│BO / RS │     │  controller  │         │  modules/    │
│(skopt) │     │  (training,  │         │  loss/       │
│        │     │   diagnosis, │         │  (flops,     │
│        │     │   tuning)    │         │   hardware)  │
└───┬────┘     └──────┬───────┘         └──────┬───────┘
    │                 │                        │
    │     ┌───────────▼─────────────┐          │
    │     │ NeuralSymbolicBridge    │◄─────────┘
    │     │ (ProbLog reasoning)     │
    │     └───────────┬─────────────┘
    │                 │
    │     ┌───────────▼─────────────┐
    │     │ tuning_rules_symbolic   │
    │     │ (Mutazione dello spazio)│
    │     └───────────┬─────────────┘
    │                 │
    ▼                 ▼
┌──────────────────────────────────────┐
│  Backend Framework (TF / PyTorch)    │
│  - NeuralNetwork (build, train)      │
│  - ModuleBackend (FLOPs, params)     │
│  - Model (architettura dinamica)     │
└──────────────────────────────────────┘
```

### Flusso di Esecuzione (Singola Iterazione)

1. **Campionamento:** BO/RS propone un vettore di iperparametri dallo spazio di ricerca corrente.
2. **Training:** Il `controller` costruisce e addestra la rete, verificando i vincoli su FLOPs/latenza.
3. **Diagnosi simbolica:** Il `NeuralSymbolicBridge` traduce le metriche numeriche in fatti Prolog, esegue l'inferenza probabilistica e identifica problemi (overfitting, underfitting, ecc.).
4. **Tuning:** Le `tuning_rules_symbolic` applicano le azioni consigliate (aggiungere/rimuovere layer, modificare i bound degli iperparametri).
5. **Aggiornamento spazio:** Lo spazio di ricerca viene espanso/restretto in base alla diagnosi.
6. **LFI:** I pesi delle regole simboliche vengono aggiornati in base al miglioramento osservato.

---

## 3. Struttura delle Directory

```
Symbolic_DNN-Tuner/
├── symbolic_tuner.py          # Entry point principale
├── exp_config.py              # Sistema di configurazione (YAML + dataclass)
├── analyze_results.py         # Analisi post-esperimento dei risultati
├── download_hf_dataset.py     # Download dataset da HuggingFace
├── test_quantizer.py          # Test del quantizzatore
├── test_roigesture_3d.py      # Test dataset ROI gesture 3D
├── roi_map_viewer.py          # Visualizzatore mappe ROI
├── requirements.txt           # Dipendenze Python
│
├── components/                # Logica core del sistema
│   ├── controller.py          # Orchestratore training/diagnosis/tuning
│   ├── search_space.py        # Gestione spazio di ricerca (skopt.Space)
│   ├── neural_network.py      # ABC per reti neurali
│   ├── model_interface.py     # Strutture dati per layer e modelli
│   ├── backend_interface.py   # ABC per backend framework
│   ├── neural_sym_bridge.py   # Ponte numerico ↔ simbolico
│   ├── objFunction.py         # Wrapper funzione obiettivo
│   ├── constraints.py         # Vincoli sullo spazio campionato
│   ├── dataset.py             # Caricamento dataset
│   ├── gesture_dataset.py     # Dataset event-camera (DVSGesture)
│   ├── improvement_checker.py # Tracciamento miglioramenti
│   ├── integral.py            # Calcolo features integrali delle curve di loss
│   ├── lfi_integration.py     # Interfaccia ProbLog LFI
│   ├── random_search.py       # Ottimizzatore Random Search
│   ├── storing_experience.py  # Persistenza esperienza (SQLite)
│   ├── tuning_rules_symbolic.py # Regole di mutazione architettura/HP
│   ├── colors.py              # Costanti colore ANSI per terminale
│   └── bo/                    # Bayesian Optimization personalizzata
│       ├── base.py            # Funzione base_minimize
│       ├── gp.py              # gp_minimize (GP-based BO)
│       └── optimizer.py       # Optimizer stateful
│
├── modules/                   # Moduli di loss pluggabili
│   ├── module.py              # Manager dei moduli
│   ├── common_interface.py    # ABC per moduli di loss
│   └── loss/                  # Moduli concreti
│       ├── flops_module.py    # Modulo FLOPs/parametri
│       ├── flops_module.pl    # Regole Prolog per FLOPs
│       ├── hardware_module.py # Modulo latenza hardware NVDLA
│       └── hardware_module.pl # Regole Prolog per hardware
│
├── pytorch_implementation/    # Implementazione PyTorch
│   ├── model.py               # Builder modello (ConvBlock, TorchModel)
│   ├── module_backend.py      # Backend FLOPs/params per PyTorch
│   └── neural_network.py      # Training loop PyTorch
│
├── tensorflow_implementation/ # Implementazione TensorFlow
│   ├── model.py               # Builder modello Keras
│   ├── module_backend.py      # Backend FLOPs/params per TensorFlow
│   ├── neural_network.py      # Training loop TensorFlow
│   └── custom_train.py        # Training personalizzato per gesture/event
│
├── symbolic_base/             # File base di ragionamento simbolico
│   ├── symbolic_analysis.pl   # Regole Prolog per diagnosi problemi
│   ├── sym_prob_base.pl       # Regole probabilistiche per azioni
│   └── lfi.pl                 # Template per Learning From Interpretations
│
├── flops/                     # Calcolo FLOPs
│   ├── flops_calculator.py    # Profiler FLOPs TensorFlow
│   └── node_manager.py        # Conversione grafo profiler in dizionario
│
├── nvdla/                     # Profilazione hardware NVDLA
│   ├── profiler.py            # Profiler latenza NVDLA
│   ├── models/                # Configurazioni hardware
│   ├── specs/                 # Specifiche acceleratore
│   └── wrapper/               # Wrapper per interfaccia C++
│
└──quantizer/                 # Quantizzazione modelli
    ├── quantizer_interface.py # ABC per quantizzatori
    ├── quantizer_POTQ.py      # Post-Training Quantization
    └── binary_quantizer.py    # Quantizzazione binaria dei pesi
```

---

## 4. Configurazione (`exp_config.py`)

Il sistema di configurazione utilizza un `dataclass` (`ConfigSchema`) con valori di default e un file YAML persistente.

### Parametri Principali

| Parametro       | Tipo      | Default             | Descrizione                                                   |
|-----------------|-----------|---------------------|---------------------------------------------------------------|
| `backend`       | `str`     | `"tf"`              | Framework: `"tf"` (TensorFlow) o `"torch"` (PyTorch)         |
| `eval`          | `int`     | `300`               | Numero massimo di valutazioni                                 |
| `early_stop`    | `int`     | `30`                | Patience per early stopping                                   |
| `epochs`        | `int`     | `2`                 | Epoche di training per valutazione                            |
| `mod_list`      | `list`    | `[]`                | Moduli attivi (es. `["flops_module", "hardware_module"]`)     |
| `dataset`       | `str`     | `"light"`           | Dataset: `cifar10`, `cifar100`, `mnist`, `tinyimagenet`, `gesture`, ecc. |
| `name`          | `str`     | `"experiment"`      | Nome esperimento (determina la directory di output)           |
| `seed`          | `int`     | `42`                | Seed per riproducibilità                                      |
| `quantization`  | `bool`    | `False`             | Abilitare la quantizzazione post-training                     |
| `verbose`       | `int`     | `2`                 | Livello di verbosità (0: silenzioso, 1: spazio, 2: spazio+modello) |
| `opt`           | `str`     | `"filtered"`        | Strategia di ottimizzazione                                   |
| `w_flops`       | `float`   | `0.33`              | Peso della loss FLOPs                                         |
| `w_HW`          | `float`   | `0.33`              | Peso della loss hardware                                      |
| `lacc`          | `float`   | `0.10`              | Soglia per underfitting: se `1 - acc > lacc`                  |
| `flops_th`      | `int`     | `150_000_000`       | Soglia massima FLOPs                                          |
| `nparams_th`    | `int`     | `2_500_000`         | Soglia massima parametri                                      |
| `frames`        | `int`     | `16`                | Numero di frame per dataset gesture                           |
| `mode`          | `str`     | `"fwdPass"`         | Modalità esperimento: `fwdPass`, `depth`, `hybrid`            |
| `channels`      | `int`     | `2`                 | Canali per dataset event-based                                |
| `polarity`      | `str`     | `"both"`            | Polarità per dataset event-based                              |
| `hf_cache`      | `str`     | `None`              | Directory cache HuggingFace                                   |

### Strategie di Ottimizzazione (`opt`)

| Strategia    | Descrizione                                                                 |
|--------------|-----------------------------------------------------------------------------|
| `standard`   | BO pura senza regole simboliche; lo spazio di ricerca è fisso               |
| `filtered`   | BO con constraint filtering + ragionamento simbolico + mutazione spazio     |
| `basic`      | BO con regole simboliche ma senza espansione dello spazio base              |
| `RS`         | Random Search pura senza regole simboliche                                  |
| `RS_ruled`   | Random Search con regole simboliche e mutazione dello spazio                |

### Ciclo di Vita della Configurazione

```python
# 1. Creazione del file YAML con override da CLI
cfg_path = create_config_file(exp_dir, overrides=args.__dict__)

# 2. Impostazione come config attiva (variabile d'ambiente)
set_active_config(cfg_path)

# 3. Caricamento (con caching thread-safe)
cfg = load_cfg(force=True)

# 4. Accesso ai parametri tramite dot-notation
print(cfg.backend)   # "tf"
print(cfg.epochs)    # 2
```

---

## 5. Entry Point Principale (`symbolic_tuner.py`)

Lo script `symbolic_tuner.py` è il punto di ingresso dell'intero sistema. La sua esecuzione segue questi passi:

### 5.1 Parsing degli Argomenti

`parse_args()` definisce i flag CLI che corrispondono ai parametri di `ConfigSchema`. Questi sovrascrivono i default nel file YAML.

### 5.2 Setup dell'Esperimento

```python
# Crea la directory e il config.yaml
cfg_path = create_config_file(exp_dir, overrides=args.__dict__)
set_active_config(cfg_path)
cfg = load_cfg(force=True)

# Import dinamico del backend
if cfg.backend == "tf":
    from tensorflow_implementation import module_backend, neural_network
elif cfg.backend == "torch":
    from pytorch_implementation import module_backend, neural_network

# Creazione directory esperimento e copia file simbolici
create_experiment_folders()
copy_symbolic_files()
```

### 5.3 Caricamento Dataset

Il `TunerDataset` carica il dataset specificato da `cfg.dataset` e lo suddivide in train/validation/test.

### 5.4 Inizializzazione Controller

Il `controller` è il cuore del sistema. Riceve la classe NN, il backend dei moduli e il dataset.

### 5.5 Loop di Ottimizzazione (`run_optimization`)

```python
while ctrl.iter <= max_iter and not ctrl.convergence:
    # 1. Campiona iperparametri con BO o RS
    res = gp_minimize(...) | random_search(...)

    # 2. Aggiorna cronologia x0, y0
    x0, y0 = res.x_iters, res.func_vals

    # 3. Diagnosi simbolica e mutazione spazio
    if cfg.opt in ["filtered", "RS_ruled", "basic"]:
        next_space = ctrl.diagnosis(const_space)
    else:
        next_space = copy.deepcopy(base_space)

    # 4. Espansione spazio se necessario
    base_space = first_ss.expand_space(base_space, next_space)

    # 5. Reset optimizer se le dimensioni cambiano
    if len(next_space.dimensions) != len(const_space.dimensions):
        x0, y0 = [], []

    const_space = copy.deepcopy(next_space)
```

---

## 6. Componenti Core (`components/`)

### 6.1 Controller

**File:** `components/controller.py`

Il `controller` è l'**orchestratore centrale** del sistema. Gestisce tre fasi principali per ogni iterazione:

- **`training(params)`**: Costruisce la rete neurale con gli iperparametri proposti, verifica i vincoli (FLOPs, latenza), esegue il training e restituisce lo score.
- **`diagnosis(const_space)`**: Calcola feature integrali dalla curva di loss, invoca il ragionamento simbolico ProbLog, identifica i problemi e propone azioni correttive.
- **`tuning(actions, space)`**: Applica le azioni di tuning (aggiunta/rimozione layer, modifica bound) allo spazio di ricerca.

Tiene traccia dello stato globale: iterazione corrente, miglior score, contatori dei layer, flag di convergenza.

### 6.2 Search Space

**File:** `components/search_space.py`

Classe `search_space` che gestisce lo spazio di ricerca basato su `skopt.Space`. Operazioni principali:

- **`search_sp(max_block, max_dense)`**: Genera lo spazio di ricerca iniziale con le dimensioni per learning rate, batch size, numero di blocchi convoluzionali, neuroni FC, dropout, ecc.
- **`add_params(space, params)`**: Aggiunge nuove dimensioni (es. quando viene aggiunto un nuovo blocco convoluzionale).
- **`remove_params(space, names)`**: Rimuove dimensioni dallo spazio.
- **`expand_space(base, new)`**: Unisce lo spazio base con quello nuovo, mantenendo le dimensioni esistenti e aggiungendo quelle nuove.

### 6.3 Objective Function e Constraints

**File:** `components/objFunction.py` e `components/constraints.py`

- **`ObjectiveWrapper`**: Adatta l'interfaccia tra l'ottimizzatore (che passa una lista di valori) e il `controller.training()` (che riceve un dizionario di parametri). Converte la lista posizionale in un dizionario nome→valore basato sulle dimensioni dello spazio corrente.

- **`ConstraintsWrapper`**: Implementa `apply_constraints()` che viene passata a `gp_minimize` come `space_constraint` nella modalità `filtered`. Filtra i campioni invalidi durante la fase di acquisizione del BO.

### 6.4 Neural Network (Abstract)

**File:** `components/neural_network.py`

Classe base astratta (`ABC`) che definisce l'interfaccia framework-agnostica per le reti neurali:

| Metodo Astratto       | Descrizione                                        |
|-----------------------|----------------------------------------------------|
| `build_network(params)` | Costruisce il modello dalla configurazione HP      |
| `training(params)`    | Esegue il ciclo di training                         |
| `save_model(path)`    | Salva il modello su disco                          |
| `eval_model(data)`    | Valuta il modello sui dati di test                 |

Le implementazioni concrete risiedono in `pytorch_implementation/` e `tensorflow_implementation/`.

### 6.5 Model Interface e Backend Interface

**File:** `components/model_interface.py` e `components/backend_interface.py`

- **`model_interface.py`** definisce:
  - `LayerSpec`: struttura dati per specificare un layer (tipo, filtri, kernel, ecc.)
  - `TunerModel`: classe base per il modello, con metodi di costruzione da un dizionario di parametri
  - `LayerTypes`: enumerazione dei tipi di layer supportati
  - `Params`: classe per gestire i parametri in modo tipizzato

- **`backend_interface.py`** definisce l'ABC `BackendInterface` per operazioni che dipendono dal framework:
  - `get_flops(model)`: calcola i FLOPs
  - `get_params(model)`: conta i parametri
  - `get_layer_info(model)`: estrae informazioni sui layer

### 6.6 Neural-Symbolic Bridge

**File:** `components/neural_sym_bridge.py`

Il `NeuralSymbolicBridge` è il **componente chiave** che collega il mondo numerico (metriche di training) al ragionamento simbolico (ProbLog):

1. **`build_symbolic_model()`**: Assembla un programma Prolog combinando:
   - Fatti numerici (accuracy, loss, FLOPs, ecc.)
   - Regole probabilistiche dal file `sym_prob_base.pl`
   - Regole di analisi da `symbolic_analysis.pl`
   - Regole dei moduli attivi (es. `flops_module.pl`)

2. **`symbolic_reasoning()`**: Esegue l'inferenza ProbLog, interrogando gli atomi `action(_, _)` e restituendo le probabilità.

3. **`edit_probs()`**: Aggiorna le probabilità delle regole in base ai pesi appresi dall'LFI (se c'è stato miglioramento).

### 6.7 Tuning Rules Symbolic

**File:** `components/tuning_rules_symbolic.py`

Contiene oltre **20 metodi** di mutazione architettura/iperparametri. Ogni metodo è un'azione che il ragionamento simbolico può raccomandare:

| Categoria       | Azioni                                                            |
|-----------------|-------------------------------------------------------------------|
| **Architettura** | `new_conv_block()`, `remove_conv_block()`, `new_fc_layer()`, `remove_fc_layer()` |
| **Regolarizzazione** | `inc_dropout()`, `decr_dropout()`, `reg_l1()`, `reg_l2()`  |
| **Learning Rate** | `decr_lr()`, `inc_lr()`                                        |
| **Batch Size**  | `inc_batch()`, `decr_batch()`                                     |
| **Data Augment** | `data_augmentation()`                                            |
| **Filtri**       | `inc_filters()`, `decr_filters()`                               |

Ogni azione modifica lo spazio di ricerca tramite `search_space.add_params()` o `search_space.remove_params()`.

### 6.8 LFI Integration

**File:** `components/lfi_integration.py`

Wrapper per il **Learning From Interpretations** di ProbLog. Impara i pesi delle regole probabilistiche dall'esperienza accumulata:

1. **`create_evidence(action, problem, improved)`**: Crea tuple di evidenza (es. *"l'azione `inc_dropout` per `overfitting` ha portato a un miglioramento"*).
2. **`learning()`**: Carica il programma LFI da `lfi.pl`, esegue `run_lfi()` con l'esperienza accumulata, ed estrae i pesi aggiornati.
3. I pesi appresi vengono passati al `NeuralSymbolicBridge` per aggiornare le probabilità delle regole.

### 6.9 Dataset e Gesture Dataset

**File:** `components/dataset.py` e `components/gesture_dataset.py`

- **`TunerDataset`**: Caricatore unificato per i dataset standard. Ogni metodo `load_*()` restituisce `(x_train, y_train, x_val, y_val, x_test, y_test)` e i metadati (`num_classes`, `input_shape`).

| Dataset         | Metodo                 | Input Shape        |
|-----------------|------------------------|--------------------|
| CIFAR-10        | `load_cifar_10()`      | (32, 32, 3)        |
| CIFAR-100       | `load_cifar_100()`     | (32, 32, 3)        |
| MNIST           | `load_mnist()`         | (28, 28, 1)        |
| TinyImageNet    | `load_tiny_imagenet()`  | (64, 64, 3)        |
| Light (subset)  | `load_light_cifar()`   | (32, 32, 3)        |
| DVSGesture      | `load_gesture()`       | (T, H, W, C)       |
| ROI Gesture     | `load_roi_gesture()`   | variabile           |

- **`gesture_dataset.py`**: Gestisce i dataset da event-camera (DVSGesture) con supporto per:
  - ROI (Region of Interest) con mappe posizionali
  - Codifica temporale one-hot (`ToOneHotTimeCoding`)
  - Trasformazioni ROI (`ROIMapTransform`)
  - Modalità: `fwdPass` (frame-by-frame), `depth` (profondità temporale), `hybrid`

### 6.10 Improvement Checker

**File:** `components/improvement_checker.py`

Classe `ImprovementChecker` che confronta lo score corrente con la cronologia per determinare se c'è stato un miglioramento. Il risultato booleano viene usato come evidenza per l'LFI.

### 6.11 Integral

**File:** `components/integral.py`

Funzione `integrals()` che calcola le feature integrali dalle curve di loss di validazione:
- **Area sotto la curva (AUC)** della loss
- **Pendenza (slope)** della loss

Queste metriche vengono tradotte in fatti Prolog (es. `slow_start`, `high_loss`) per il ragionamento simbolico.

### 6.12 Random Search

**File:** `components/random_search.py`

Classe `RandomSearch` come alternativa alla Bayesian Optimization. Campiona uniformemente dallo spazio di ricerca. Utilizzata quando `cfg.opt` contiene `"RS"`.

### 6.13 Storing Experience

**File:** `components/storing_experience.py`

Classe `StoringExperience` che implementa un database SQLite per la persistenza dell'esperienza di tuning:
- `insert_ranking(score, params)`: Salva il risultato di ogni iterazione
- `insert_evidence(action, problem, improved)`: Salva le evidenze per LFI
- `get()`: Recupera le evidenze storiche

### 6.14 Bayesian Optimization (`bo/`)

**File:** `components/bo/base.py`, `components/bo/gp.py`, `components/bo/optimizer.py`

Implementazione personalizzata della Bayesian Optimization basata su `scikit-optimize`:

- **`gp.py`**: Espone `gp_minimize()` con supporto per `space_constraint` (usata nella modalità `filtered`)
- **`base.py`**: Funzione `base_minimize()` che gestisce il loop interno BO
- **`optimizer.py`**: Classe `Optimizer` stateful per utilizzo avanzato

---

## 7. Moduli di Loss (`modules/`)

### 7.1 Module Manager

**File:** `modules/module.py`

La classe `module` gestisce il caricamento dinamico e l'orchestrazione dei moduli di loss:

- **`load_modules(mod_list)`**: Carica dinamicamente i moduli specificati in `cfg.mod_list`
- **`get_rules()`**: Raccoglie le regole Prolog da tutti i moduli attivi
- **`state()`**: Aggiorna lo stato di tutti i moduli con le metriche correnti
- **`values()`**: Restituisce i fatti Prolog da tutti i moduli
- **`optimization()`**: Calcola il contributo di loss aggregato di tutti i moduli

### 7.2 Common Interface

**File:** `modules/common_interface.py`

ABC che definisce il contratto per tutti i moduli di loss:

```python
class common_interface(ABC):
    facts: List[str]           # Nomi dei fatti Prolog esportati
    problems: List[str]        # Nomi dei problemi diagnosticabili
    weight: float              # Peso nel contributo alla loss totale

    @abstractmethod
    def update_state(self, *args) -> None: ...
    def obtain_values(self) -> Dict[str, Any]: ...
    def optimiziation_function(self) -> float: ...
    def printing_values(self) -> None: ...
    def log_function(self) -> None: ...
```

### 7.3 FLOPs Module

**File:** `modules/loss/flops_module.py` + `modules/loss/flops_module.pl`

Modulo per la gestione dei vincoli su FLOPs e numero di parametri:

- **Python**: Calcola il gap tra FLOPs/parametri correnti e le soglie, restituisce una penalità scalare
- **Prolog**: Definisce regole probabilistiche che collegano i fatti (es. `flops_over_threshold`) ai problemi (es. `latency`, `model_size`)

### 7.4 Hardware Module

**File:** `modules/loss/hardware_module.py` + `modules/loss/hardware_module.pl`

Modulo per la profilazione della latenza su acceleratore NVDLA:

- **Python**: Chiama il profiler NVDLA per stimare latenza e costi energetici su 3 configurazioni hardware
- **Prolog**: Collega la latenza hardware al problema `out_range`

---

## 8. Implementazioni Framework-Specific

### 8.1 PyTorch

**Directory:** `pytorch_implementation/`

| File                | Descrizione                                                            |
|---------------------|------------------------------------------------------------------------|
| `model.py`          | Definisce `ConvBlock` (Conv→BN→ReLU→Pool→Dropout) e `TorchModel` che estende `nn.Module` e `TunerModel`. L'architettura è costruita dinamicamente dal dizionario di parametri. |
| `module_backend.py` | Implementa `BackendInterface` per PyTorch. Usa `FlopCounterMode` per il conteggio preciso dei FLOPs. |
| `neural_network.py` | Implementa il ciclo di training PyTorch con early stopping, salvataggio del miglior modello e logging. |

### 8.2 TensorFlow

**Directory:** `tensorflow_implementation/`

| File                | Descrizione                                                            |
|---------------------|------------------------------------------------------------------------|
| `model.py`          | Definisce `TFModel` basato su Keras Sequential/Functional API. Supporta connessioni residuali e input duali per dataset ROI. |
| `module_backend.py` | Implementa `BackendInterface` per TensorFlow, delegando al `flops_calculator`.  |
| `neural_network.py` | Implementa il ciclo di training Keras. Per dataset gesture, delega a `custom_train.py`. |
| `custom_train.py`   | Loop di training personalizzato per dataset temporali: forward pass frame-by-frame con majority voting per l'evaluazione. Supporta le modalità `fwdPass`, `depth` e `hybrid`. |

---

## 9. Ragionamento Simbolico (`symbolic_base/`)

Il sistema utilizza **ProbLog** per il ragionamento probabilistico. I file `.pl` definiscono:

### `symbolic_analysis.pl`

Regole Prolog per la **diagnosi dei problemi** di training a partire dalle metriche:

```prolog
% Esempio: diagnosi di overfitting
problem(overfitting) :- gap_tr_te_acc.
gap_tr_te_acc :- a(A), va(VA), last(A,LTA), last(VA,ScoreA), abs2(LTA,ScoreA,Gap), Gap > 0.1.

% Esempio: diagnosi di slow start
problem(slow_start) :- integral_high, slope_low.
```

### `sym_prob_base.pl`

Regole **probabilistiche** che associano le azioni ai problemi diagnosticati:

```prolog
0.50::action(reg_l2, overfitting) :- problem(overfitting).
0.99::action(data_augmentation, underfitting) :- problem(underfitting).
0.70::action(inc_dropout, overfitting) :- problem(overfitting).
```

Le probabilità (es. `0.50`) sono i pesi iniziali che vengono aggiornati dall'LFI.

### `lfi.pl`

Template per le probabilità **apprendibili** in formato LFI di ProbLog:

```prolog
t(0.50)::action(inc_dropout, overfitting).
t(0.70)::action(decr_lr, slow_convergence).
```

Il prefisso `t(...)` indica che il valore è un parametro apprendibile.

---

## 10. FLOPs & Profilazione Hardware

### 10.1 Calculator FLOPs (`flops/`)

- **`flops_calculator.py`**: Classe `flop_calculator` che utilizza il profiler TensorFlow per calcolare i FLOPs di un modello. Metodo principale: `get_flops(model)`.
- **`node_manager.py`**: Utility `to_dict()` che converte ricorsivamente il grafo del profiler TF in un dizionario operazione → FLOPs.

### 10.2 Profiler NVDLA (`nvdla/`)

- **`profiler.py`**: Classe `nvdla` che modella l'acceleratore NVIDIA NVDLA. Carica le configurazioni hardware da file YAML e calcola i tempi di esecuzione per le operazioni:
  - `getCONVTime()`: Tempo di convoluzione
  - `getSDPTime()`: Tempo SDP (normalization, activation)
  - `getPDPTime()`: Tempo pooling

Le configurazioni hardware supportate definiscono diverse architetture NVDLA con parametri come MAC array size, bandwidth, clock frequency.

---

## 11. Quantizzazione (`quantizer/`)

Sistema opzionale di quantizzazione post-training, attivabile con `--quantization`:

| File                    | Descrizione                                                      |
|-------------------------|------------------------------------------------------------------|
| `quantizer_interface.py` | ABC per quantizzatori: `quantizer_function()`, `evaluate_quantized_model()`, `save_quantized_model()` |
| `quantizer_POTQ.py`    | Post-Training Quantization (quantizzazione dei pesi e delle attivazioni) |
| `binary_quantizer.py`  | Quantizzazione binaria dei pesi (1 bit)                          |

---

## 12. Script di Utilità

| Script                   | Descrizione                                                                  |
|--------------------------|------------------------------------------------------------------------------|
| `analyze_results.py`    | Analizza i log degli esperimenti completati. Crea CSV per esperimento e un CSV riassuntivo con il miglior risultato per ciascuno. Legge da `algorithm_logs/`, `config.yaml`, e i log dei moduli. |
| `multi_sbatch.py`       | Genera file `params.txt` con tutte le combinazioni di parametri per i job array SLURM. Supporta configurazioni per CIFAR e gesture. |
| `download_hf_dataset.py` | Scarica dataset da HuggingFace (es. TinyImageNet, DVSGesture) nella cache locale. |
| `roi_map_viewer.py`     | Visualizza le mappe ROI per dataset gesture/event-based.                     |
| `test_quantizer.py`     | Script di test per il modulo di quantizzazione.                              |
| `test_roigesture_3d.py` | Script di test per il dataset ROI gesture 3D.                                |

---

## 13. Esecuzione su Cluster HPC (SLURM)

Il progetto include script SLURM per l'esecuzione su cluster HPC:

### Esecuzione Singola (`run.slurm`)

```bash
#!/bin/sh
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00

python symbolic_tuner.py --name test_dataset --dataset tinyimagenet \
    --eval 2 --epochs 3 --backend tf
```

### Job Array — Dataset Immagine (`jonarray.slurm`)

Esegue combinazioni di dataset × ottimizzatore × seed da `params.txt`:

```bash
#SBATCH --array=0-4%4
# Parametri letti da params.txt: DATA,OPT,SEED

python symbolic_tuner.py \
  --name "results_FLOPS/${NAME_EXP}" \
  --dataset "$DATA" --opt "$OPT" --seed "$SEED" \
  --eval 1000 --epochs 100 --mod_list flops_module
```

### Job Array — Dataset Gesture (`jonarray_gesture.slurm`)

Esegue combinazioni di modalità × frame × canali da `params_gesture.txt`:

```bash
#SBATCH --array=0-16%4
# Parametri: DATA,MODE,FRAMES,CHANNELS,PARAMS,MODULE

python symbolic_tuner.py \
  --name "results_gesture_new/${NAME_EXP}" \
  --dataset "$DATA" --mode "$MODE" \
  --frames "$FRAMES" --channels "$CHANNELS" \
  --opt filtered --seed 42 --eval 1000 --epochs 100
```

### Esempio di Utilizzo Locale

```bash
# Esperimento minimo su CIFAR-10
python symbolic_tuner.py \
  --name my_experiment \
  --dataset cifar10 \
  --backend torch \
  --opt filtered \
  --eval 50 \
  --epochs 10 \
  --mod_list flops_module \
  --seed 42

# Analisi dei risultati
python analyze_results.py
```

**Parametri comuni**:
- `--name`: Identificatore univoco per l'esperimento (usato per il nome della directory)
- `--dataset`: Dataset da usare (default: "light")
- `--backend`: Framework da usare (tf o torch)
- `--opt`: Strategia di ottimizzazione (standard, filtered, basic, RS, RS_ruled)
- `--eval`: Numero massimo di valutazioni
- `--epochs`: Numero di epoche di training per valutazione
- `--mod_list`: Lista di moduli attivi (flops_module, hardware_module)

---

## 14. Struttura dei Log di Algoritmo

La directory **`algorithm_logs/`** contiene un insieme completo di file che registrano il comportamento e i risultati dell'algoritmo di ottimizzazione simbolico durante l'esecuzione di un esperimento. Questi file consentono l'analisi dettagliata del processo di tuning, la ricostruzione della storia di ottimizzazione e l'apprendimento del sistema.

### 14.1 File di Log

#### 14.1.1 `hyper-neural.txt` — Spazio Iperparametrico

**Descrizione**: Contiene un dizionario Python per ogni iterazione di ottimizzazione, registrando i valori degli iperparametri testati.

**Formato**: Una lista Python per riga, dove ogni elemento è un dizionario con chiavi:
- `num_neurons`: numero di neuroni nei layer densi (es. 8, 16, 32, 64)
- `unit_c1`, `unit_c2`: numero di filtri nei blocchi convoluzionali
- `dr_f`: dropout rate nel layer denso (es. 0.1-0.8)
- `learning_rate`: tasso di apprendimento (es. 0.0001-0.01)
- `batch_size`: dimensione del batch (es. 16, 32, 64)
- `optimizer`: tipo di ottimizzatore (SGD, Adam, Adamax, Adadelta, Adagrad, RMSprop)
- `activation`: funzione di attivazione (relu, elu, swish, selu)
- `data_augmentation`: 0 o 1 (booleano)
- `reg_l2`: regolarizzazione L2 (0 o valore > 0)
- `skip_connection`: 0 o 1 (booleano per residual connections)
- `new_fc_X`: numero di layer densi aggiuntivi

**Esempio**:
```python
# Iterazione 1
{'num_neurons': 32, 'unit_c1': 4, 'unit_c2': 8, 'dr_f': 0.5, 'learning_rate': 0.001, 'batch_size': 32, 'optimizer': 'adam', 'activation': 'relu', 'data_augmentation': 0, 'reg_l2': 0.0001, 'skip_connection': 0, 'new_fc_1': 0}

# Iterazione 2
{'num_neurons': 24, 'unit_c1': 3, 'unit_c2': 6, 'dr_f': 0.3, 'learning_rate': 0.0005, 'batch_size': 16, 'optimizer': 'sgd', 'activation': 'elu', 'data_augmentation': 1, 'reg_l2': 0.0, 'skip_connection': 1, 'new_fc_1': 1}
```

#### 14.1.2 `acc_report.txt` — Report di Accuratezza

**Descrizione**: Contiene il valore di accuratezza nel validation set per ogni iterazione dell'ottimizzazione.

**Formato**: Un numero float per riga (range 0.0-1.0), ordinati cronologicamente.

**Intervallo tipico**: Nei primi step esperimenti mostrano accuratezza 0.4-0.6 (underfitting), con progressione a 0.8-0.95 negli step finali.

**Esempio**:
```
0.485
0.512
0.548
0.629
0.734
0.823
0.891
0.934
```

#### 14.1.3 `score_report.txt` — Score dell'Obiettivo

**Descrizione**: Registra il valore della funzione obiettivo (negato per compatibilità con maximizer) per ogni iterazione.

**Formato**: Un numero float negativo per riga, dove il valore assoluto rappresenta l'accuratezza negata per l'ottimizzazione.

**Utilizzo**: Utilizzato da scikit-optimize per guida bayesiana; la relazione con `acc_report.txt` è: `score = -accuracy` se non sono presenti moduli.

**Esempio**:
```
-0.485
-0.512
-0.548
-0.629
-0.734
-0.823
-0.891
-0.934
```

#### 14.1.4 `params_report.txt` — Parametri di Rete

**Descrizione**: Registra il numero di parametri (o FLOPs) della rete neurale risultante per ogni iterazione.

**Formato**: Un numero intero per riga, rappresentante il numero totale di parametri o operazioni floating-point.

**Intervallo tipico**: Da 1-10 milioni (reti piccole) a 100-200 milioni (reti grandi) in base alla configurazione sperimentale.

**Utilizzo**: Correlazione con accuratezza; generalmente reti più grandi hanno accuratezza migliore ma costo computazionale più alto.

**Esempio**:
```
1250000
2100000
3450000
5680000
8900000
12300000
16750000
24500000
```

#### 14.1.5 `diagnosis_symbolic_logs.txt` — Diagnosi Simboliche

**Descrizione**: Contiene le diagnosi simboliche generate dal modello probabilistico ProbLog, identificando i problemi di training nella rete neurale.

**Formato**: Una lista Python di stringhe per riga, dove ogni stringa è un tipo di diagnosi.

**Tipi di diagnosi**:
- `underfitting`: La rete ha capacità insufficiente per il compito (accuratezza < threshold)
- `overfitting`: Alta accuratezza di training ma bassa di validation (gap importante)
- `floating_loss`: Loss non converge o fluttua (problema di training)
- `need_skip`: Suggerimento di aggiungere skip connections per migliorare flusso del gradiente

**Evoluzione tipica**: Primi step dominati da `underfitting`, transizione a `overfitting` negli step finali, presenza occasionale di `floating_loss` in configurazioni problematiche.

**Esempio**:
```python
['underfitting']
['underfitting', 'floating_loss']
['underfitting']
['underfitting']
['overfitting']
['overfitting', 'need_skip']
['floating_loss']
['overfitting']
```

#### 14.1.6 `tuning_symbolic_logs.txt` — Azioni di Tuning Proposte

**Descrizione**: Contiene le azioni di tuning simboliche proposte dal sistema per affrontare le diagnosi identified in ogni iterazione.

**Formato**: Una lista Python di stringhe per riga, dove ogni stringa è un tipo di azione di tuning.

**Tipi di azioni**:
- `data_augmentation`: Aggiungere data augmentation per combattere overfitting
- `add_residual`: Aggiungere skip connections
- `inc_conv_layers`: Incrementare numero di layer convoluzionali
- `inc_batch_size`: Aumentare batch size
- `decr_lr`: Diminuire learning rate
- `dec_dropout`: Decrementare dropout
- `new_fc_layers`: Aggiungere layer densi aggiuntivi
- `new_conv_block`: Aggiungere blocchi convoluzionali
- `inc_neurons`: Incrementare numero di neuroni
- `reg_l2`: Applicare regolarizzazione L2
- `remove_reg_l2`: Rimuovere regolarizzazione L2

**Evoluzione**: Primi step actions singole (es. `['inc_neurons']`), step intermedi con 2 azioni (es. `['data_augmentation', 'add_residual']`), step finali con 2-3 azioni complesse.

**Esempio**:
```python
['inc_neurons']
['inc_neurons', 'inc_conv_layers']
['inc_batch_size', 'decr_lr']
['data_augmentation', 'add_residual']
['dec_dropout']
['data_augmentation', 'inc_neurons', 'add_residual']
['decr_lr']
['new_fc_layers', 'dec_dropout']
```

#### 14.1.7 `evidence.txt` — Evidenza per Apprendimento da Interpretazioni (LFI)

**Descrizione**: Registra coppie (azione, risultato) che documentano l'efficacia di ogni azione di tuning intrapresa. Utilizzato dal sistema Learning From Interpretations (LFI) per adattare i pesi delle regole simboliche.

**Formato**: Una tupla Python per riga con struttura: `[(action(metodo, diagnosi), successo), ...]`

**Campo `successo`**: 
- `True`: L'azione ha risolto la diagnosi (accuratezza migliorata)
- `False`: L'azione non ha risolto la diagnosi (accuratezza non migliorata)

**Utilizzo**: L'algoritmo di LFI analizza questo file per imparare quali combinazioni (azione, diagnosi) sono più efficaci, adattando i pesi delle regole probabilistiche per future iterazioni.

**Esempio**:
```python
[(action(inc_neurons, underfitting), False), (action(inc_conv_layers, underfitting), False)]
[(action(inc_batch_size, floating_loss), True), (action(decr_lr, floating_loss), True)]
[(action(data_augmentation, overfitting), False)]
[(action(add_residual, need_skip), True), (action(dec_dropout, overfitting), False)]
```

### 14.2 Come Utilizzare i Log

#### Analisi Python

```python
import json
import ast

# Percorso dell'esperimento
result_dir = "results_gesture/gesture/26_03_20_15_75562_gesture_hybrid_64_8_BASELINE"
log_dir = f"{result_dir}/algorithm_logs"

# Carica file
hyper_params = []
with open(f"{log_dir}/hyper-neural.txt", "r") as f:
    for line in f:
        hyper_params.append(ast.literal_eval(line.strip()))

accuracies = []
with open(f"{log_dir}/acc_report.txt", "r") as f:
    accuracies = [float(line.strip()) for line in f]

diagnoses = []
with open(f"{log_dir}/diagnosis_symbolic_logs.txt", "r") as f:
    for line in f:
        diagnoses.append(ast.literal_eval(line.strip()))

actions = []
with open(f"{log_dir}/tuning_symbolic_logs.txt", "r") as f:
    for line in f:
        actions.append(ast.literal_eval(line.strip()))

evidence = []
with open(f"{log_dir}/evidence.txt", "r") as f:
    for line in f:
        evidence.append(ast.literal_eval(line.strip()))

# Analizza correlazione tra azioni e miglioramento di accuratezza
for i, (hyp, acc, diag, act) in enumerate(zip(hyper_params, accuracies, diagnoses, actions)):
    if i > 0:
        acc_improvement = accuracies[i] - accuracies[i-1]
        print(f"Iterazione {i}: Accuratezza={acc:.3f} (+{acc_improvement:.3f}), "
              f"Ottimizzatore={hyp['optimizer']}, Azioni={act}")
        if 'underfitting' in diag:
            print(f"  → Diagnosi: UNDERFITTING")
        elif 'overfitting' in diag:
            print(f"  → Diagnosi: OVERFITTING")
```

#### Estrazione di Statistiche

```python
# Numero di iterazioni
n_iters = len(accuracies)
print(f"Iterazioni totali: {n_iters}")

# Accuratezza iniziale e finale
print(f"Accuratezza iniziale: {accuracies[0]:.3f}")
print(f"Accuratezza finale: {accuracies[-1]:.3f}")
print(f"Miglioramento: {accuracies[-1] - accuracies[0]:.3f}")

# Diagnosi più frequenti
from collections import Counter
all_diagnoses = []
for diag_list in diagnoses:
    all_diagnoses.extend(diag_list)

diag_counter = Counter(all_diagnoses)
print(f"Frequenza diagnosi: {dict(diag_counter)}")

# Azioni più frequenti
all_actions = []
for action_list in actions:
    all_actions.extend(action_list)

action_counter = Counter(all_actions)
print(f"Frequenza azioni: {dict(action_counter)}")
```

### 14.3 Struttura Completa della Directory

La struttura tipica di una directory di risultati è la seguente:

```
results_gesture/
├── gesture/
│   └── 26_03_20_15_75562_gesture_hybrid_64_8_BASELINE/  # Identificativo esperimento
│       ├── algorithm_logs/  # Directory principale dei log
│       │   ├── hyper-neural.txt          # 28 iperparametri (una riga per iterazione)
│       │   ├── acc_report.txt            # 28 accuratezze (0.485 → 0.934)
│       │   ├── score_report.txt          # 28 score negativi (-0.934 → -0.485)
│       │   ├── params_report.txt         # 28 conteggi parametri network
│       │   ├── diagnosis_symbolic_logs.txt # 28 diagnosi symboliche
│       │   ├── tuning_symbolic_logs.txt   # 28 azioni di tuning proposte
│       │   └── evidence.txt               # 27 coppie (azione, risultato)
│       ├── best_models/
│       │   ├── best_accuracy_model.pth    # Miglior modello per accuratezza
│       │   └── best_flops_model.pth       # Miglior modello per FLOPs (se multi-obiettivo)
│       ├── config.yaml                    # Configurazione dell'esperimento
│       ├── results_summary.txt             # Riassunto finale accuratezza/FLOPs
│       └── training_logs.txt               # Log di training di ciascun step
```

**Note sulla Numerazione**:
- A ogni iterazione di ottimizzazione bayesiana, una nuova riga viene aggiunta ai 7 file log
- `evidence.txt` contiene n-1 righe (genera l'evidenza DOPO il test della configurazione)
- File sono **text-based** per facilità di parsing e debugging manuale

---

## 15. Interfaccia Grafica (GUI)

### Panoramica

Il sistema Symbolic DNN-Tuner include un'interfaccia grafica interattiva costruita con **Streamlit** che consente di analizzare e visualizzare i risultati degli esperimenti di tuning. La GUI è accessibile tramite il file principale `GUI.py` e fornisce un'esperienza utente intuitiva per l'analisi post-esperimento.

**File correlati**:
- `GUI.py`: Script principale dell'interfaccia Streamlit
- `graphfun.py`: Funzioni di visualizzazione e creazione grafici (Plotly)
- `analyze_out.py`: Funzioni di analisi e caricamento dati

### 15.1 Avvio della GUI

Per lanciare l'interfaccia grafica:

```bash
# Avvia l'applicazione Streamlit
streamlit run GUI.py

# L'applicazione si aprirà nel browser al seguente indirizzo:
# http://localhost:8501
```

**Requisiti**:
- `streamlit>=1.40.0`
- `plotly>=5.0.0`
- I file di log degli esperimenti (directory `algorithm_logs/`)

### 15.2 Fasi Operative della GUI

La GUI è organizzata in **4 fasi principali**:

#### Fase 1: Selezione Cartella (Folder Selection)

**Funzione**: `folder_selection_phase()`

Consente all'utente di selezionare le cartelle di input e output:

- **Cartella da analizzare**: Cartella contenente i risultati degli esperimenti (con subdirectory `algorithm_logs/`)
- **Cartella di salvataggio (opzionale)**: Directory dove salvare i grafici generati

**Caratteristiche**:
- Supporto sia per dialog file picker (su Windows/Linux) che per input manuale di path
- Gestione sicura su macOS dove Tkinter può causare problemi
- Validazione del percorso inserito
- Informazione visiva del percorso selezionato

#### Fase 2: Caricamento Dati (Loading Phase)

**Funzione**: `loading_phase()`

Carica gli esperimenti dalla cartella selezionata e permette la scelta dei grafici:

- **Caricamento esperimenti**: Utilizza `analyze_all_experiments()` per leggere i log
- **Selezione esperimenti**: Multiselect per scegliere quali esperimenti analizzare
- **Selezione grafici**: Tre modalità di visualizzazione:
  1. **Individual charts**: Grafici per singoli esperimenti
  2. **Comparison charts**: Grafici comparativi tra più esperimenti
  3. **Search space**: Visualizzazione dello spazio di ricerca

**Grafici disponibili**:
- Line plot Accuracy, Score e Parametri
- Bar plot efficacia azioni
- Bar plot diagnosi simboliche
- Bar plot azioni di tuning
- Timeline scatter plot azioni/diagnosi

#### Fase 3: Analisi (Analysis Phase)

**Funzione**: `analysis_phase()`

Genera e visualizza i grafici selezionati:

- **Rendering interattivo**: Utilizzando Plotly con selezione punti
- **Detail modal**: Cliccando su punti nel grafico di accuratezza si aprono dettagli per quella iterazione
- **Salvataggio**: Opzione di salvare i singoli grafici o tutti insieme
- **Interattività**: Tutti i grafici supportano zoom, pan e altre interazioni Plotly

#### Fase 4: Completamento (Final Phase)

**Funzione**: Gestita in `main()`

Schermata finale di successo con:
- Confermé del completamento
- Animazione celebrativa (balloons)
- Opzione di tornare alla fase iniziale per una nuova analisi

### 15.3 Classi Principali

#### `ExperimentResult`

Dataclass che rappresenta una singola iterazione di un esperimento:

```python
@dataclass
class ExperimentResult:
    iteration: int                          # Numero dell'iterazione
    accuracy: Optional[float]               # Accuratezza di validazione
    flops: Optional[float]                  # Numero di FLOPs
    nparams: Optional[float]                # Numero di parametri
    latency: Optional[float]                # Latenza di inferenza
    hw_cost: Optional[float]                # Costo hardware (energia)
    hw_total_cost: Optional[float]          # Costo totale hardware
    hw_config: Optional[str]                # Configurazione hardware
    evidence: Optional[Tuple[Tuple[str, str], bool]]  # Evidenza LFI
    diagnosis: Optional[List[str]]          # Diagnosi simboliche trovate
    tuning: Optional[List[str]]             # Azioni di tuning applicate
    score: Optional[float] = None           # Score (calcolato come -accuracy)
```

#### `ResultsAnalyzer`

Classe per l'analisi dei risultati degli esperimenti:

**Metodi principali**:
- `load_results()`: Carica tutti i log dall'esperimento
- `load_config()`: Legge la configurazione dell'esperimento da `config.yaml`
- `get_results()`: Restituisce la lista di `ExperimentResult`
- `analyze_evidence()`: Analizza l'efficacia delle azioni di tuning

**Attributi**:
- `experiment_dir`: Percorso dell'esperimento
- `algorithm_logs_dir`: Percorso della directory dei log
- `results`: Lista di risultati caricati
- `has_flops_module`: Flag se il modulo FLOPs è presente
- `has_hardware_module`: Flag se il modulo hardware è presente
- `exp_name`: Nome dell'esperimento

### 15.4 Funzioni di Visualizzazione (graphfun.py)

Le funzioni principali di Plotly disponibili sono:

| Funzione | Descrizione |
|----------|-------------|
| `plotaccuracy()` | Grafico a linee per accuratezza, score e parametri di un esperimento |
| `plotaccuracy_confronto()` | Versione comparativa tra più esperimenti |
| `plotevidence()` | Bar plot dell'efficacia delle azioni di tuning |
| `plotevidence_confronto()` | Versione comparativa |
| `plotdiagnosis()` | Bar plot delle diagnosi simboliche trovate |
| `plotdiagnosis_confronto()` | Versione comparativa |
| `plottuning()` | Bar plot delle azioni di tuning applicate |
| `plottuning_confronto()` | Versione comparativa |
| `plottimeline()` | Timeline scatter plot azioni vs diagnosi |
| `plottimeline_confronto()` | Versione comparativa |

**Funzionalità comuni**:
- Supporto per salvataggio in PNG/PDF
- Interattività con Plotly (zoom, pan, hover info)
- Legenda personalizzata
- Annotazioni per valori significativi
- Colori standardizzati per leggibilità

### 15.5 Funzioni di Analisi (analyze_out.py)

Funzioni per l'analisi e il caricamento dati:

| Funzione | Descrizione |
|----------|-------------|
| `analyze_all_experiments()` | Carica tutti gli esperimenti da una directory e restituisce analyzer |
| `find_out()` | Trova i file di output e crea oggetti per lo spazio di ricerca |
| `load_experiment()` | Carica un singolo esperimento |

### 15.6 Stato Streamlit (Session State)

La GUI utilizza il sistema di session state di Streamlit per mantenere lo stato tra i rerun:

```python
st.session_state.fase                       # Fase attuale (selezione_cartella, caricamento, analisi, fine)
st.session_state.cartellaInput              # Cartella da analizzare
st.session_state.cartellaOutput             # Cartella dove salvare i grafici
st.session_state.lista_esperimenti          # Lista di esperimenti caricati
st.session_state.lista_analizer             # Lista di analyzer per gli esperimenti
st.session_state.lista_grafici_singoli      # Grafici da generare per singoli esperimenti
st.session_state.lista_grafici_confronto    # Grafici da generare per confronti
st.session_state.stato_bottone_singoli      # Flag per modalità individual charts
st.session_state.stato_bottone_confronto    # Flag per modalità comparison charts
st.session_state.stato_bottone_spazio_ricerca  # Flag per modalità search space
```

### 15.7 Gestione File Dialog

La GUI gestisce le differenze tra sistemi operativi per il file picker:

- **Windows/Linux**: Utilizza dialog Tkinter standard
- **macOS**: Richiede attenzione al thread principale; fallback a input manuale se necessario
- **Validazione**: Verifica che il percorso esista e sia una directory

```python
def _is_macos():
    """Verifica se il sistema è macOS"""
    return platform.system().lower() == "darwin"

def _ask_directory_safe():
    """Apre il file dialog in modo sicuro"""
    if not _can_use_tk_dialog():
        return None
    # ... usa Tkinter filedialog
```

### 15.8 Modalità di Salvataggio

#### Salvataggio Singolo

Ogni grafico ha un pulsante "Save" che permette il salvataggio individuale:

```python
if st.button(f"Save {grafico}", key=f"salva_singolo_{grafico}"):
    if plotaccuracy(analizer, esperimento, st.session_state.cartellaOutput) is None:
        st.warning(f"Could not save {grafico}")
    else:
        st.success(f"Successfully saved {grafico}!")
```

#### Salvataggio Bulk

Pulsante "Save all" salva contemporaneamente tutti i grafici selezionati:

```python
if st.button('Save all'):
    # Salva tutti i grafici in una sola operazione
    st.session_state.salvato_tutto = True
    st.rerun()
```

### 15.9 Dialoghi Informativi

La GUI include dialoghi popup per le descrizioni dei grafici:

```python
@st.dialog("ℹ️ Chart description")
def mostra_descrizione(testo):
    st.write(testo)
```

**Informazioni disponibili**:
- Descrizione di ogni tipo di grafico
- Interpretazione dei risultati
- Suggerimenti per l'analisi

---

## 16. Debug Mode

### Definizione

Il **Debug Mode** è una modalità operativa che permette di testare la pipeline di tuning senza eseguire il training effettivo della rete neurale. Quando il nome dell'esperimento contiene la parola **"debug"** (case-insensitive), il sistema genera valori casuali di loss e accuracy per ogni epoca, simulando un training completo.

### Utilizzo

Per attivare il Debug Mode, includere la parola "debug" nel nome dell'esperimento:

```bash
# Attiva Debug Mode
python symbolic_tuner.py --name debug_test_experiment --backend tf --epoch 10 --eval 5

# Attiva Debug Mode col backend PyTorch
python symbolic_tuner.py --name debug_torch_test --backend torch --epoch 10 --eval 5

# Disattiva Debug Mode (niente "debug" nel nome)
python symbolic_tuner.py --name experiment_v1 --backend tf --epoch 10 --eval 5
```

### Implementazione per Backend

#### PyTorch (`pytorch_implementation/neural_network.py`)

Il check del debug mode avviene **dentro il loop di epoch**, permettendo alternanza tra debug e training reale:

```python
for epoch in range(self.exp_cfg.epochs):
    if 'debug' in self.exp_cfg.name.lower():
        # Genera valori casuali per questa epoca
        train_loss = float(np.random.uniform(0.5, 2.0))
        val_loss = float(np.random.uniform(0.5, 2.0))
        train_acc = float(np.random.uniform(0.4, 0.95))
        val_acc = float(np.random.uniform(0.4, 0.95))
    else:
        # Esegue il training reale
        self.model.train()
        # ... forward pass, backward pass, evaluation
```

**Vantaggi**:
- ✅ Flessibilità: Permette test rapidi senza perdere la struttura
- ✅ Overhead minimo: Solo un if per epoca
- ✅ Mantiene la compatibility con early stopping e learning rate scheduling

**Output console**:
```
Epoch 1: loss=1.2345, val_loss=1.5678, acc=0.6234, val_acc=0.5890
Epoch 2: loss=1.0987, val_loss=1.4321, acc=0.6789, val_acc=0.6234
```

#### TensorFlow (`tensorflow_implementation/neural_network.py`)

Il check del debug mode avviene **prima del training**, creando due branch separati:

```python
if 'debug' in self.exp_cfg.name.lower():
    # Genera tutte le epoche con valori casuali
    for epoch in range(self.epochs):
        train_loss = float(np.random.uniform(0.5, 2.0))
        # ... genera valori e stampa nel formato TensorFlow
else:
    # Esegue il training reale
    history = self.model.model.fit(...)
```

**Vantaggi**:
- ✅ Chiarezza strutturale: Separazione netta tra debug e training
- ✅ Realismo output: Include batch count e timing metriche
- ✅ Completezza: Gestisce tutti i parametri di configurazione

**Output console** (formato TensorFlow reale):
```
Epoch 1/100
17/17 - 0s - loss: 1.2345 - accuracy: 0.6234 - val_loss: 1.5678 - val_accuracy: 0.5890 - 175ms/epoch - 10ms/step
Epoch 2/100
17/17 - 0s - loss: 1.0987 - accuracy: 0.6789 - val_loss: 1.4321 - val_accuracy: 0.6234 - 180ms/epoch - 11ms/step
```

### Distribuzione dei Valori Casuali

Entrambe le implementazioni usano `np.random.uniform()` con i seguenti intervalli:

| Parametro | Min | Max | Motivo |
|-----------|-----|-----|--------|
| Loss (train/val) | 0.5 | 2.0 | Range realistico per classificazione su CIFAR-10/ImageNet |
| Accuracy (train/val) | 0.4 | 0.95 | Range corrispondente a early-stage training |
| Epoch time (ms) | 150 | 200 | Simulazione realistica di timing hardware |

### Setup Configurato Automaticamente

**Prima** del debug check, entrambe le implementazioni configurano comunque:

1. **Ottimizzatore** (SGD, Adam, RMSprop)
2. **Loss function** (CrossEntropyLoss per PyTorch, categorical_crossentropy per TF)
3. **Learning rate scheduler** (ReduceLROnPlateau) – anche se non utilizzato
4. **Data loaders/preparation** (DataLoader per PyTorch, X_train/X_test per TF)

Questo garantisce che il `model.optimizer` sia sempre disponibile per il controller, anche in debug mode.

### Casi d'Uso Principali

Il Debug Mode è utile per:

- 🧪 **Test rapidi della pipeline**: Verificare il flusso senza attendere il training
- ⚡ **Validazione della struttura del codice**: Controllare che logging, callbacks e risultati funzionino
- 📊 **Verifica della logica di diagnosi**: Testare il raginonamento simbolico e le azioni proposte
- 🔍 **Debug del controller**: Verificare le operazioni senza attendere epoche di training
- 📈 **Profiling dell'infrastruttura**: Analizzare overhead di logging, saving, etc. senza calcolo pesante


---

## 17. Dipendenze

```
tensorflow==2.15          # Backend TF (opzionale se si usa PyTorch)
scikit-optimize>=0.10.2   # Bayesian Optimization
matplotlib>=3.9.0         # Plotting
pytest>=8.3.4             # Testing
tonic>=1.5.0              # Dataset event-camera (DVSGesture)
problog>=2.2.6            # Ragionamento simbolico probabilistico
tqdm==4.67.1              # Barre di progresso
PyYAML==6.0.3             # Configurazione
pandas==2.3.3             # Analisi risultati
streamlit>=1.40.0         # Interfaccia grafica web
plotly>=5.0.0             # Visualizzazione interattiva grafici
```

Per PyTorch, è necessario installare separatamente:
```
torch>=2.0
torchvision
```

---

## 18. Riferimenti Bibliografici

1. Fraccaroli, M., Lamma, E., & Riguzzi, F. (2022). *Symbolic DNN-tuner*. Machine Learning, 111(2), 625–650. [DOI: 10.1007/s10994-021-06097-1](https://link.springer.com/article/10.1007/s10994-021-06097-1)

2. Fraccaroli, M., Lamma, E., & Riguzzi, F. (2022). *Symbolic DNN-Tuner: A Python and ProbLog-based system for optimizing Deep Neural Networks hyperparameters*. SoftwareX, 17, 100957. [DOI: 10.1016/j.softx.2021.100957](https://www.sciencedirect.com/science/article/pii/S2352711021001825)
