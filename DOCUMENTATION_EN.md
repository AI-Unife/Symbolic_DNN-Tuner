# Symbolic DNN-Tuner — Technical Documentation

> **Original Authors:** Michele Fraccaroli, Evelina Lamma, Fabrizio Riguzzi  
> **Repository:** [micheleFraccaroli/Symbolic_DNN-Tuner](https://github.com/micheleFraccaroli/Symbolic_DNN-Tuner)  
> **Active Branch:** `main_gesture`

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [System Architecture](#2-system-architecture)
3. [Directory Structure](#3-directory-structure)
4. [Configuration (`exp_config.py`)](#4-configuration-exp_configpy)
5. [Main Entry Point (`symbolic_tuner.py`)](#5-main-entry-point-symbolic_tunerpy)
6. [Core Components (`components/`)](#6-core-components-components)
   - 6.1 [Controller](#61-controller)
   - 6.2 [Search Space](#62-search-space)
   - 6.3 [Objective Function and Constraints](#63-objective-function-and-constraints)
   - 6.4 [Neural Network (Abstract)](#64-neural-network-abstract)
   - 6.5 [Model Interface and Backend Interface](#65-model-interface-and-backend-interface)
   - 6.6 [Neural-Symbolic Bridge](#66-neural-symbolic-bridge)
   - 6.7 [Tuning Rules Symbolic](#67-tuning-rules-symbolic)
   - 6.8 [LFI Integration](#68-lfi-integration)
   - 6.9 [Dataset and Gesture Dataset](#69-dataset-and-gesture-dataset)
   - 6.10 [Improvement Checker](#610-improvement-checker)
   - 6.11 [Integral](#611-integral)
   - 6.12 [Random Search](#612-random-search)
   - 6.13 [Storing Experience](#613-storing-experience)
   - 6.14 [Bayesian Optimization (`bo/`)](#614-bayesian-optimization-bo)
7. [Loss Modules (`modules/`)](#7-loss-modules-modules)
   - 7.1 [Module Manager](#71-module-manager)
   - 7.2 [Common Interface](#72-common-interface)
   - 7.3 [FLOPs Module](#73-flops-module)
   - 7.4 [Hardware Module](#74-hardware-module)
8. [Framework-Specific Implementations](#8-framework-specific-implementations)
   - 8.1 [PyTorch](#81-pytorch)
   - 8.2 [TensorFlow](#82-tensorflow)
9. [Symbolic Reasoning (`symbolic_base/`)](#9-symbolic-reasoning-symbolic_base)
10. [FLOPs & Hardware Profiling](#10-flops--hardware-profiling)
    - 10.1 [FLOPs Calculator (`flops/`)](#101-flops-calculator-flops)
    - 10.2 [NVDLA Profiler (`nvdla/`)](#102-nvdla-profiler-nvdla)
11. [Quantization (`quantizer/`)](#11-quantization-quantizer)
12. [Utility Scripts](#12-utility-scripts)
13. [HPC Cluster Execution (SLURM)](#13-hpc-cluster-execution-slurm)
14. [Design Patterns](#14-design-patterns)
15. [Dependencies](#15-dependencies)
16. [References](#16-references)

---

## 1. Project Overview

**Symbolic DNN-Tuner** is an automatic hyperparameter optimization system (AutoML) for deep neural networks that combines:

- **Bayesian Optimization (BO)** or **Random Search (RS)** for hyperparameter sampling;
- **Symbolic reasoning** via ProbLog for intelligent diagnosis of training problems and proposal of corrective actions;
- **Learning From Interpretations (LFI)** for probabilistic adaptation of symbolic rule weights based on accumulated experience;
- **Modular loss functions** for multi-objective optimization (accuracy, FLOPs, hardware latency).

The system supports two deep learning backends (TensorFlow and PyTorch), handles standard datasets (CIFAR-10/100, MNIST, TinyImageNet) and event-based gesture datasets (DVSGesture), and includes hardware profiling on NVDLA and optional quantization.

---

## 2. System Architecture

```
┌─────────────────────────────────────────────────────────┐
│                   symbolic_tuner.py                      │
│                (Main Orchestrator)                        │
└────────────────────────┬────────────────────────────────┘
                         │
         ┌───────────────▼────────────────┐
         │     exp_config.py              │
         │  (YAML Configuration)          │
         └───────────────┬────────────────┘
                         │
         ┌───────────────▼──────────────────────────────┐
         │          OPTIMIZATION LOOP                    │
         │     (max_iter iterations or convergence)      │
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
    │     │ (Space mutation)        │
    │     └───────────┬─────────────┘
    │                 │
    ▼                 ▼
┌──────────────────────────────────────┐
│  Backend Framework (TF / PyTorch)    │
│  - NeuralNetwork (build, train)      │
│  - ModuleBackend (FLOPs, params)     │
│  - Model (dynamic architecture)      │
└──────────────────────────────────────┘
```

### Execution Flow (Single Iteration)

1. **Sampling:** BO/RS proposes a hyperparameter vector from the current search space.
2. **Training:** The `controller` builds and trains the network, checking FLOPs/latency constraints.
3. **Symbolic Diagnosis:** The `NeuralSymbolicBridge` translates numerical metrics into Prolog facts, runs probabilistic inference, and identifies problems (overfitting, underfitting, etc.).
4. **Tuning:** The `tuning_rules_symbolic` apply recommended actions (add/remove layers, modify hyperparameter bounds).
5. **Space Update:** The search space is expanded/contracted based on the diagnosis.
6. **LFI:** Symbolic rule weights are updated based on observed improvement.

---

## 3. Directory Structure

```
Symbolic_DNN-Tuner/
├── symbolic_tuner.py          # Main entry point
├── exp_config.py              # Configuration system (YAML + dataclass)
├── multi_sbatch.py            # Parameter generator for SLURM job arrays
├── analyze_results.py         # Post-experiment result analysis
├── download_hf_dataset.py     # HuggingFace dataset downloader
├── test_quantizer.py          # Quantizer test script
├── test_roigesture_3d.py      # ROI gesture 3D dataset test
├── roi_map_viewer.py          # ROI map viewer
├── requirements.txt           # Python dependencies
│
├── components/                # Core system logic
│   ├── controller.py          # Training/diagnosis/tuning orchestrator
│   ├── search_space.py        # Search space management (skopt.Space)
│   ├── neural_network.py      # ABC for neural networks
│   ├── model_interface.py     # Layer and model data structures
│   ├── backend_interface.py   # ABC for framework backends
│   ├── neural_sym_bridge.py   # Numeric ↔ symbolic bridge
│   ├── objFunction.py         # Objective function wrapper
│   ├── constraints.py         # Sampled space constraints
│   ├── dataset.py             # Dataset loading
│   ├── gesture_dataset.py     # Event-camera dataset (DVSGesture)
│   ├── improvement_checker.py # Improvement tracking
│   ├── integral.py            # Integral features from loss curves
│   ├── lfi_integration.py     # ProbLog LFI interface
│   ├── random_search.py       # Random Search optimizer
│   ├── storing_experience.py  # Experience persistence (SQLite)
│   ├── tuning_rules_symbolic.py # Architecture/HP mutation rules
│   ├── colors.py              # ANSI terminal color constants
│   └── bo/                    # Custom Bayesian Optimization
│       ├── base.py            # base_minimize function
│       ├── gp.py              # gp_minimize (GP-based BO)
│       └── optimizer.py       # Stateful optimizer
│
├── modules/                   # Pluggable loss modules
│   ├── module.py              # Module manager
│   ├── common_interface.py    # ABC for loss modules
│   └── loss/                  # Concrete modules
│       ├── flops_module.py    # FLOPs/parameters module
│       ├── flops_module.pl    # Prolog rules for FLOPs
│       ├── hardware_module.py # NVDLA hardware latency module
│       └── hardware_module.pl # Prolog rules for hardware
│
├── pytorch_implementation/    # PyTorch implementation
│   ├── model.py               # Model builder (ConvBlock, TorchModel)
│   ├── module_backend.py      # PyTorch FLOPs/params backend
│   └── neural_network.py      # PyTorch training loop
│
├── tensorflow_implementation/ # TensorFlow implementation
│   ├── model.py               # Keras model builder
│   ├── module_backend.py      # TensorFlow FLOPs/params backend
│   ├── neural_network.py      # TensorFlow training loop
│   └── custom_train.py        # Custom training for gesture/event data
│
├── symbolic_base/             # Base symbolic reasoning files
│   ├── symbolic_analysis.pl   # Prolog rules for problem diagnosis
│   ├── sym_prob_base.pl       # Probabilistic rules for actions
│   └── lfi.pl                 # Learning From Interpretations template
│
├── flops/                     # FLOPs calculation
│   ├── flops_calculator.py    # TensorFlow FLOPs profiler
│   └── node_manager.py        # Profiler graph to dictionary converter
│
├── nvdla/                     # NVDLA hardware profiling
│   ├── profiler.py            # NVDLA latency profiler
│   ├── profilerEMBER.py       # EMBER-integrated profiler
│   ├── models/                # Hardware configurations
│   ├── specs/                 # Accelerator specifications
│   └── wrapper/               # C++ interface wrapper
│
└── quantizer/                 # Model quantization
    ├── quantizer_interface.py # ABC for quantizers
    ├── quantizer_POTQ.py      # Post-Training Quantization
    └── binary_quantizer.py    # Binary weight quantization
```

---

## 4. Configuration (`exp_config.py`)

The configuration system uses a `dataclass` (`ConfigSchema`) with default values and a persistent YAML file.

### Main Parameters

| Parameter       | Type      | Default             | Description                                                     |
|-----------------|-----------|---------------------|-----------------------------------------------------------------|
| `backend`       | `str`     | `"tf"`              | Framework: `"tf"` (TensorFlow) or `"torch"` (PyTorch)          |
| `eval`          | `int`     | `300`               | Maximum number of evaluations                                   |
| `early_stop`    | `int`     | `30`                | Early stopping patience                                         |
| `epochs`        | `int`     | `2`                 | Training epochs per evaluation                                  |
| `mod_list`      | `list`    | `[]`                | Active modules (e.g., `["flops_module", "hardware_module"]`)    |
| `dataset`       | `str`     | `"light"`           | Dataset: `cifar10`, `cifar100`, `mnist`, `tinyimagenet`, `gesture`, etc. |
| `name`          | `str`     | `"experiment"`      | Experiment name (determines output directory)                   |
| `seed`          | `int`     | `42`                | Seed for reproducibility                                        |
| `quantization`  | `bool`    | `False`             | Enable post-training quantization                               |
| `verbose`       | `int`     | `2`                 | Verbosity level (0: silent, 1: space, 2: space+model)           |
| `opt`           | `str`     | `"filtered"`        | Optimization strategy                                           |
| `w_flops`       | `float`   | `0.33`              | FLOPs loss weight                                               |
| `w_HW`          | `float`   | `0.33`              | Hardware loss weight                                            |
| `lacc`          | `float`   | `0.10`              | Underfitting threshold: if `1 - acc > lacc`                     |
| `flops_th`      | `int`     | `150_000_000`       | Maximum FLOPs threshold                                         |
| `nparams_th`    | `int`     | `2_500_000`         | Maximum parameters threshold                                    |
| `frames`        | `int`     | `16`                | Number of frames for gesture dataset                            |
| `mode`          | `str`     | `"fwdPass"`         | Experiment mode: `fwdPass`, `depth`, `hybrid`                   |
| `channels`      | `int`     | `2`                 | Channels for event-based datasets                               |
| `polarity`      | `str`     | `"both"`            | Polarity for event-based datasets                               |
| `hf_cache`      | `str`     | `None`              | HuggingFace cache directory                                     |

### Optimization Strategies (`opt`)

| Strategy     | Description                                                                   |
|--------------|-------------------------------------------------------------------------------|
| `standard`   | Pure BO without symbolic rules; the search space is fixed                     |
| `filtered`   | BO with constraint filtering + symbolic reasoning + space mutation            |
| `basic`      | BO with symbolic rules but without base space expansion                       |
| `RS`         | Pure Random Search without symbolic rules                                     |
| `RS_ruled`   | Random Search with symbolic rules and space mutation                          |

### Configuration Lifecycle

```python
# 1. Create YAML file with CLI overrides
cfg_path = create_config_file(exp_dir, overrides=args.__dict__)

# 2. Set as active config (environment variable)
set_active_config(cfg_path)

# 3. Load (with thread-safe caching)
cfg = load_cfg(force=True)

# 4. Access parameters via dot-notation
print(cfg.backend)   # "tf"
print(cfg.epochs)    # 2
```

---

## 5. Main Entry Point (`symbolic_tuner.py`)

The `symbolic_tuner.py` script is the entry point for the entire system. Its execution follows these steps:

### 5.1 Argument Parsing

`parse_args()` defines CLI flags that correspond to `ConfigSchema` parameters. These override the defaults in the YAML file.

### 5.2 Experiment Setup

```python
# Create the directory and config.yaml
cfg_path = create_config_file(exp_dir, overrides=args.__dict__)
set_active_config(cfg_path)
cfg = load_cfg(force=True)

# Dynamic backend import
if cfg.backend == "tf":
    from tensorflow_implementation import module_backend, neural_network
elif cfg.backend == "torch":
    from pytorch_implementation import module_backend, neural_network

# Create experiment directories and copy symbolic files
create_experiment_folders()
copy_symbolic_files()
```

### 5.3 Dataset Loading

The `TunerDataset` loads the dataset specified by `cfg.dataset` and splits it into train/validation/test.

### 5.4 Controller Initialization

The `controller` is the heart of the system. It receives the NN class, the module backend, and the dataset.

### 5.5 Optimization Loop (`run_optimization`)

```python
while ctrl.iter <= max_iter and not ctrl.convergence:
    # 1. Sample hyperparameters with BO or RS
    res = gp_minimize(...) | random_search(...)

    # 2. Update history x0, y0
    x0, y0 = res.x_iters, res.func_vals

    # 3. Symbolic diagnosis and space mutation
    if cfg.opt in ["filtered", "RS_ruled", "basic"]:
        next_space = ctrl.diagnosis(const_space)
    else:
        next_space = copy.deepcopy(base_space)

    # 4. Expand space if necessary
    base_space = first_ss.expand_space(base_space, next_space)

    # 5. Reset optimizer if dimensions change
    if len(next_space.dimensions) != len(const_space.dimensions):
        x0, y0 = [], []

    const_space = copy.deepcopy(next_space)
```

---

## 6. Core Components (`components/`)

### 6.1 Controller

**File:** `components/controller.py`

The `controller` is the **central orchestrator** of the system. It manages three main phases per iteration:

- **`training(params)`**: Builds the neural network with the proposed hyperparameters, checks constraints (FLOPs, latency), runs training, and returns the score.
- **`diagnosis(const_space)`**: Computes integral features from the loss curve, invokes ProbLog symbolic reasoning, identifies problems, and proposes corrective actions.
- **`tuning(actions, space)`**: Applies tuning actions (add/remove layers, modify bounds) to the search space.

It tracks global state: current iteration, best score, layer counters, and convergence flag.

### 6.2 Search Space

**File:** `components/search_space.py`

The `search_space` class manages the search space based on `skopt.Space`. Main operations:

- **`search_sp(max_block, max_dense)`**: Generates the initial search space with dimensions for learning rate, batch size, number of convolutional blocks, FC neurons, dropout, etc.
- **`add_params(space, params)`**: Adds new dimensions (e.g., when a new convolutional block is added).
- **`remove_params(space, names)`**: Removes dimensions from the space.
- **`expand_space(base, new)`**: Merges the base space with the new one, keeping existing dimensions and adding new ones.

### 6.3 Objective Function and Constraints

**Files:** `components/objFunction.py` and `components/constraints.py`

- **`ObjectiveWrapper`**: Adapts the interface between the optimizer (which passes a list of values) and `controller.training()` (which receives a parameter dictionary). Converts the positional list to a name→value dictionary based on the current space dimensions.

- **`ConstraintsWrapper`**: Implements `apply_constraints()` which is passed to `gp_minimize` as `space_constraint` in `filtered` mode. Filters invalid samples during the BO acquisition phase.

### 6.4 Neural Network (Abstract)

**File:** `components/neural_network.py`

Abstract base class (`ABC`) that defines the framework-agnostic interface for neural networks:

| Abstract Method        | Description                                     |
|------------------------|-------------------------------------------------|
| `build_network(params)` | Builds the model from the HP configuration     |
| `training(params)`     | Runs the training loop                           |
| `save_model(path)`     | Saves the model to disk                          |
| `eval_model(data)`     | Evaluates the model on test data                 |

Concrete implementations reside in `pytorch_implementation/` and `tensorflow_implementation/`.

### 6.5 Model Interface and Backend Interface

**Files:** `components/model_interface.py` and `components/backend_interface.py`

- **`model_interface.py`** defines:
  - `LayerSpec`: data structure for specifying a layer (type, filters, kernel, etc.)
  - `TunerModel`: base class for the model, with methods for construction from a parameter dictionary
  - `LayerTypes`: enumeration of supported layer types
  - `Params`: class for typed parameter management

- **`backend_interface.py`** defines the `BackendInterface` ABC for framework-dependent operations:
  - `get_flops(model)`: computes FLOPs
  - `get_params(model)`: counts parameters
  - `get_layer_info(model)`: extracts layer information

### 6.6 Neural-Symbolic Bridge

**File:** `components/neural_sym_bridge.py`

The `NeuralSymbolicBridge` is the **key component** connecting the numerical world (training metrics) to symbolic reasoning (ProbLog):

1. **`build_symbolic_model()`**: Assembles a Prolog program by combining:
   - Numerical facts (accuracy, loss, FLOPs, etc.)
   - Probabilistic rules from `sym_prob_base.pl`
   - Analysis rules from `symbolic_analysis.pl`
   - Active module rules (e.g., `flops_module.pl`)

2. **`symbolic_reasoning()`**: Runs ProbLog inference, querying `action(_, _)` atoms and returning probabilities.

3. **`edit_probs()`**: Updates rule probabilities based on weights learned by LFI (if improvement occurred).

### 6.7 Tuning Rules Symbolic

**File:** `components/tuning_rules_symbolic.py`

Contains over **20 methods** for architecture/hyperparameter mutation. Each method is an action that symbolic reasoning can recommend:

| Category          | Actions                                                              |
|-------------------|----------------------------------------------------------------------|
| **Architecture**  | `new_conv_block()`, `remove_conv_block()`, `new_fc_layer()`, `remove_fc_layer()` |
| **Regularization** | `inc_dropout()`, `decr_dropout()`, `reg_l1()`, `reg_l2()`          |
| **Learning Rate** | `decr_lr()`, `inc_lr()`                                             |
| **Batch Size**    | `inc_batch()`, `decr_batch()`                                       |
| **Data Augment**  | `data_augmentation()`                                                |
| **Filters**       | `inc_filters()`, `decr_filters()`                                   |

Each action modifies the search space via `search_space.add_params()` or `search_space.remove_params()`.

### 6.8 LFI Integration

**File:** `components/lfi_integration.py`

Wrapper for ProbLog's **Learning From Interpretations**. Learns probabilistic rule weights from accumulated experience:

1. **`create_evidence(action, problem, improved)`**: Creates evidence tuples (e.g., *"the action `inc_dropout` for `overfitting` led to improvement"*).
2. **`learning()`**: Loads the LFI program from `lfi.pl`, runs `run_lfi()` with accumulated experience, and extracts updated weights.
3. Learned weights are passed to `NeuralSymbolicBridge` to update rule probabilities.

### 6.9 Dataset and Gesture Dataset

**Files:** `components/dataset.py` and `components/gesture_dataset.py`

- **`TunerDataset`**: Unified loader for standard datasets. Each `load_*()` method returns `(x_train, y_train, x_val, y_val, x_test, y_test)` and metadata (`num_classes`, `input_shape`).

| Dataset         | Method                  | Input Shape        |
|-----------------|-------------------------|--------------------|
| CIFAR-10        | `load_cifar_10()`       | (32, 32, 3)        |
| CIFAR-100       | `load_cifar_100()`      | (32, 32, 3)        |
| MNIST           | `load_mnist()`          | (28, 28, 1)        |
| TinyImageNet    | `load_tiny_imagenet()`  | (64, 64, 3)        |
| Light (subset)  | `load_light_cifar()`    | (32, 32, 3)        |
| DVSGesture      | `load_gesture()`        | (T, H, W, C)       |
| ROI Gesture     | `load_roi_gesture()`    | variable            |

- **`gesture_dataset.py`**: Handles event-camera datasets (DVSGesture) with support for:
  - ROI (Region of Interest) with positional maps
  - One-hot temporal encoding (`ToOneHotTimeCoding`)
  - ROI transformations (`ROIMapTransform`)
  - Modes: `fwdPass` (frame-by-frame), `depth` (temporal depth), `hybrid`

### 6.10 Improvement Checker

**File:** `components/improvement_checker.py`

The `ImprovementChecker` class compares the current score against history to determine whether improvement has occurred. The boolean result is used as evidence for LFI.

### 6.11 Integral

**File:** `components/integral.py`

The `integrals()` function computes integral features from validation loss curves:
- **Area Under the Curve (AUC)** of the loss
- **Slope** of the loss

These metrics are translated into Prolog facts (e.g., `slow_start`, `high_loss`) for symbolic reasoning.

### 6.12 Random Search

**File:** `components/random_search.py`

The `RandomSearch` class serves as an alternative to Bayesian Optimization. It samples uniformly from the search space. Used when `cfg.opt` contains `"RS"`.

### 6.13 Storing Experience

**File:** `components/storing_experience.py`

The `StoringExperience` class implements a SQLite database for tuning experience persistence:
- `insert_ranking(score, params)`: Saves the result of each iteration
- `insert_evidence(action, problem, improved)`: Saves evidence for LFI
- `get()`: Retrieves historical evidence

### 6.14 Bayesian Optimization (`bo/`)

**Files:** `components/bo/base.py`, `components/bo/gp.py`, `components/bo/optimizer.py`

Custom Bayesian Optimization implementation based on `scikit-optimize`:

- **`gp.py`**: Exposes `gp_minimize()` with support for `space_constraint` (used in `filtered` mode)
- **`base.py`**: `base_minimize()` function that manages the inner BO loop
- **`optimizer.py`**: Stateful `Optimizer` class for advanced usage

---

## 7. Loss Modules (`modules/`)

### 7.1 Module Manager

**File:** `modules/module.py`

The `module` class manages dynamic loading and orchestration of loss modules:

- **`load_modules(mod_list)`**: Dynamically loads modules specified in `cfg.mod_list`
- **`get_rules()`**: Collects Prolog rules from all active modules
- **`state()`**: Updates the state of all modules with current metrics
- **`values()`**: Returns Prolog facts from all modules
- **`optimization()`**: Computes the aggregate loss contribution from all modules

### 7.2 Common Interface

**File:** `modules/common_interface.py`

ABC that defines the contract for all loss modules:

```python
class common_interface(ABC):
    facts: List[str]           # Names of exported Prolog facts
    problems: List[str]        # Names of diagnosable problems
    weight: float              # Weight in the total loss contribution

    @abstractmethod
    def update_state(self, *args) -> None: ...
    def obtain_values(self) -> Dict[str, Any]: ...
    def optimiziation_function(self) -> float: ...
    def printing_values(self) -> None: ...
    def log_function(self) -> None: ...
```

### 7.3 FLOPs Module

**Files:** `modules/loss/flops_module.py` + `modules/loss/flops_module.pl`

Module for managing FLOPs and parameter count constraints:

- **Python**: Computes the gap between current FLOPs/parameters and thresholds, returns a scalar penalty
- **Prolog**: Defines probabilistic rules linking facts (e.g., `flops_over_threshold`) to problems (e.g., `latency`, `model_size`)

### 7.4 Hardware Module

**Files:** `modules/loss/hardware_module.py` + `modules/loss/hardware_module.pl`

Module for NVDLA accelerator latency profiling:

- **Python**: Calls the NVDLA profiler to estimate latency and energy costs across 3 hardware configurations
- **Prolog**: Links hardware latency to the `out_range` problem

---

## 8. Framework-Specific Implementations

### 8.1 PyTorch

**Directory:** `pytorch_implementation/`

| File                | Description                                                            |
|---------------------|------------------------------------------------------------------------|
| `model.py`          | Defines `ConvBlock` (Conv→BN→ReLU→Pool→Dropout) and `TorchModel` extending `nn.Module` and `TunerModel`. Architecture is dynamically built from the parameter dictionary. |
| `module_backend.py` | Implements `BackendInterface` for PyTorch. Uses `FlopCounterMode` for precise FLOPs counting. |
| `neural_network.py` | Implements the PyTorch training loop with early stopping, best model saving, and logging. |

### 8.2 TensorFlow

**Directory:** `tensorflow_implementation/`

| File                | Description                                                            |
|---------------------|------------------------------------------------------------------------|
| `model.py`          | Defines `TFModel` based on Keras Sequential/Functional API. Supports residual connections and dual inputs for ROI datasets. |
| `module_backend.py` | Implements `BackendInterface` for TensorFlow, delegating to `flops_calculator`. |
| `neural_network.py` | Implements the Keras training loop. For gesture datasets, delegates to `custom_train.py`. |
| `custom_train.py`   | Custom training loop for temporal datasets: frame-by-frame forward pass with majority voting for evaluation. Supports `fwdPass`, `depth`, and `hybrid` modes. |

---

## 9. Symbolic Reasoning (`symbolic_base/`)

The system uses **ProbLog** for probabilistic reasoning. The `.pl` files define:

### `symbolic_analysis.pl`

Prolog rules for **diagnosing training problems** from metrics:

```prolog
% Example: overfitting diagnosis
problem(overfitting) :- gap_tr_te_acc.
gap_tr_te_acc :- a(A), va(VA), last(A,LTA), last(VA,ScoreA), abs2(LTA,ScoreA,Gap), Gap > 0.1.

% Example: slow start diagnosis
problem(slow_start) :- integral_high, slope_low.
```

### `sym_prob_base.pl`

**Probabilistic rules** associating actions with diagnosed problems:

```prolog
0.50::action(reg_l2, overfitting) :- problem(overfitting).
0.99::action(data_augmentation, underfitting) :- problem(underfitting).
0.70::action(inc_dropout, overfitting) :- problem(overfitting).
```

The probabilities (e.g., `0.50`) are initial weights that get updated by LFI.

### `lfi.pl`

Template for **learnable probabilities** in ProbLog's LFI format:

```prolog
t(0.50)::action(inc_dropout, overfitting).
t(0.70)::action(decr_lr, slow_convergence).
```

The `t(...)` prefix indicates that the value is a learnable parameter.

---

## 10. FLOPs & Hardware Profiling

### 10.1 FLOPs Calculator (`flops/`)

- **`flops_calculator.py`**: The `flop_calculator` class uses the TensorFlow profiler to compute model FLOPs. Main method: `get_flops(model)`.
- **`node_manager.py`**: Utility `to_dict()` that recursively converts the TF profiler graph into an operation → FLOPs dictionary.

### 10.2 NVDLA Profiler (`nvdla/`)

- **`profiler.py`**: The `nvdla` class models the NVIDIA NVDLA accelerator. It loads hardware configurations from YAML files and computes execution times for operations:
  - `getCONVTime()`: Convolution time
  - `getSDPTime()`: SDP time (normalization, activation)
  - `getPDPTime()`: Pooling time

- **`profilerEMBER.py`**: Version integrated with the EMBER framework for profiling on custom accelerator specifications.

Supported hardware configurations define different NVDLA architectures with parameters such as MAC array size, bandwidth, and clock frequency.

---

## 11. Quantization (`quantizer/`)

Optional post-training quantization system, enabled with `--quantization`:

| File                    | Description                                                        |
|-------------------------|--------------------------------------------------------------------|
| `quantizer_interface.py` | ABC for quantizers: `quantizer_function()`, `evaluate_quantized_model()`, `save_quantized_model()` |
| `quantizer_POTQ.py`    | Post-Training Quantization (weight and activation quantization)    |
| `binary_quantizer.py`  | Binary weight quantization (1 bit)                                  |

---

## 12. Utility Scripts

| Script                   | Description                                                                    |
|--------------------------|--------------------------------------------------------------------------------|
| `analyze_results.py`    | Analyzes logs from completed experiments. Creates per-experiment CSVs and a summary CSV with the best result for each. Reads from `algorithm_logs/`, `config.yaml`, and module logs. |
| `multi_sbatch.py`       | Generates `params.txt` files with all parameter combinations for SLURM job arrays. Supports CIFAR and gesture configurations. |
| `download_hf_dataset.py` | Downloads datasets from HuggingFace (e.g., TinyImageNet, DVSGesture) to local cache. |
| `roi_map_viewer.py`     | Visualizes ROI maps for gesture/event-based datasets.                          |
| `test_quantizer.py`     | Test script for the quantization module.                                        |
| `test_roigesture_3d.py` | Test script for the ROI gesture 3D dataset.                                     |

---

## 13. HPC Cluster Execution (SLURM)

The project includes SLURM scripts for execution on HPC clusters:

### Single Execution (`run.slurm`)

```bash
#!/bin/sh
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00

python symbolic_tuner.py --name test_dataset --dataset tinyimagenet \
    --eval 2 --epochs 3 --backend tf
```

### Job Array — Image Datasets (`jonarray.slurm`)

Runs combinations of dataset × optimizer × seed from `params.txt`:

```bash
#SBATCH --array=0-4%4
# Parameters read from params.txt: DATA,OPT,SEED

python symbolic_tuner.py \
  --name "results_FLOPS/${NAME_EXP}" \
  --dataset "$DATA" --opt "$OPT" --seed "$SEED" \
  --eval 1000 --epochs 100 --mod_list flops_module
```

### Job Array — Gesture Datasets (`jonarray_gesture.slurm`)

Runs combinations of mode × frames × channels from `params_gesture.txt`:

```bash
#SBATCH --array=0-16%4
# Parameters: DATA,MODE,FRAMES,CHANNELS,PARAMS,MODULE

python symbolic_tuner.py \
  --name "results_gesture_new/${NAME_EXP}" \
  --dataset "$DATA" --mode "$MODE" \
  --frames "$FRAMES" --channels "$CHANNELS" \
  --opt filtered --seed 42 --eval 1000 --epochs 100
```

### Local Usage Example

```bash
# Minimal experiment on CIFAR-10
python symbolic_tuner.py \
  --name my_experiment \
  --dataset cifar10 \
  --backend torch \
  --opt filtered \
  --eval 50 \
  --epochs 10 \
  --mod_list flops_module \
  --seed 42

# Analyze results
python analyze_results.py
```

---

## 14. Design Patterns

| Pattern              | Where                                       | Purpose                                          |
|----------------------|---------------------------------------------|--------------------------------------------------|
| **Strategy**         | Backend (TF/PyTorch)                        | Framework interchangeability via `BackendInterface` |
| **Template Method**  | `NeuralNetwork` ABC                         | Subclasses implement `build_network()`, `training()` |
| **Factory**          | `module.load_modules()`                     | Dynamic loading of loss modules                   |
| **Bridge**           | `NeuralSymbolicBridge`                      | Connecting numerical world ↔ Prolog reasoning     |
| **Adapter**          | `ObjectiveWrapper`                          | Converts positional list ↔ parameter dictionary   |
| **Command**          | `tuning_rules_symbolic`                     | Each action is an independently invocable method  |
| **Observer**         | Callbacks (EarlyStopping, logging)          | Notifies events during training                   |

---

## 15. Dependencies

```
tensorflow==2.15          # TF backend (optional if using PyTorch)
scikit-optimize>=0.10.2   # Bayesian Optimization
matplotlib>=3.9.0         # Plotting
pytest>=8.3.4             # Testing
tonic>=1.5.0              # Event-camera datasets (DVSGesture)
problog>=2.2.6            # Probabilistic symbolic reasoning
tqdm==4.67.1              # Progress bars
PyYAML==6.0.3             # Configuration
pandas==2.3.3             # Result analysis
```

For PyTorch, install separately:
```
torch>=2.0
torchvision
```

---

## 16. References

1. Fraccaroli, M., Lamma, E., & Riguzzi, F. (2022). *Symbolic DNN-tuner*. Machine Learning, 111(2), 625–650. [DOI: 10.1007/s10994-021-06097-1](https://link.springer.com/article/10.1007/s10994-021-06097-1)

2. Fraccaroli, M., Lamma, E., & Riguzzi, F. (2022). *Symbolic DNN-Tuner: A Python and ProbLog-based system for optimizing Deep Neural Networks hyperparameters*. SoftwareX, 17, 100957. [DOI: 10.1016/j.softx.2021.100957](https://www.sciencedirect.com/science/article/pii/S2352711021001825)
