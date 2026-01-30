# Symbolic DNN Tuner - Technical Documentation

## Table of Contents

1. [Overview](#overview)
2. [Architecture](#architecture)
3. [Core Components](#core-components)
4. [Symbolic Reasoning System](#symbolic-reasoning-system)
5. [Training Pipeline](#training-pipeline)
6. [Search Space Management](#search-space-management)
7. [Configuration System](#configuration-system)
8. [Usage Examples](#usage-examples)
9. [Extending the Framework](#extending-the-framework)

---

## Overview



The **Symbolic DNN Tuner** is a neuro-symbolic framework for automated deep learning model optimization. It combines Bayesian optimization, symbolic reasoning via ProbLog, and Learning From Interpretations (LFI) to iteratively tune neural network architectures and hyperparameters.

### Key Features

- **Neuro-Symbolic Integration**: Bridges numeric training metrics with probabilistic logic programming
- **Automated Architecture Search**: Dynamically adds/removes layers based on performance
- **Multi-Objective Optimization**: Balances accuracy, FLOPs, latency, and model size
- **Hardware-Aware Tuning**: Considers deployment constraints (NVDLA accelerators, energy)
- **Interpretable Decisions**: Symbolic rules explain why modifications are proposed

### Supported Datasets

- CIFAR-10/100
- Tiny ImageNet
- Light version of CIFAR-10

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                         Main Loop                           │
│  (main.py → run_optimization → controller.training)         │
└──────────────┬──────────────────────────────────────────────┘
               │
       ┌───────▼────────┐
       │   Controller   │
       │  (Orchestrator)│
       └───────┬────────┘
               │
   ┌───────────┼───────────┐
   │           │           │
┌──▼──┐   ┌────▼───┐  ┌────▼────┐
│ NN  │   │ Search │  │Symbolic │
│Build│   │ Space  │  │Reasoning│
└─────┘   └────────┘  └─────────┘
               │           │
          ┌────▼───────────▼────┐
          │  Bayesian Optimizer │
          │   (gp_minimize)     │
          └─────────────────────┘
```

### Data Flow

1. **Initialization**: Load dataset → Create search space → Initialize modules
2. **Optimization Loop**:
   - Sample hyperparameters (BO or Random Search)
   - Build & train network
   - Evaluate performance + constraints
   - Symbolic diagnosis (ProbLog)
   - Propose repairs (LFI-learned probabilities)
   - Update search space
3. **Termination**: Convergence or max evaluations reached

---

## Core Components

### 1. Controller (`components/controller.py`)

**Purpose**: Orchestrates the entire tuning process.

**Key Responsibilities**:

- Manages training lifecycle
- Bridges neural networks ↔ symbolic reasoning
- Tracks best models and convergence
- Coordinates modules (FLOPs, hardware)

**Critical Methods**:

```python
def training(self, params: Dict[str, Any]) -> float:
    """
    Builds network, trains, evaluates.
    Returns: optimization score (lower = better)
    """

def diagnosis(self, const_space) -> Space:
    """
    Runs symbolic reasoning to identify issues.
    Returns: updated search space
    """
```

**Convergence Criteria**:

- No improvement for `cfg.early_stop` iterations (default: 30)
- Max evaluations (`cfg.eval`) reached
- Tuning rules trigger `self.tr.count_no_probs > 5`

---

### 2. Neural Network (`components/neural_network.py`)

**Architecture Template**:

```python
inputs → [Data Aug?] → Conv(c1) + Act + BN → MaxPool
      → Conv(c2) + Act + BN → MaxPool
      → [N × Dynamic Conv Blocks] → GlobalAvgPool
      → [M × Dense Layers] → Dropout → Output(softmax)
```

**Dynamic Components**:

- `new_conv_N`: Additional conv blocks (controlled by search space)
- `new_fc_N`: Additional dense layers
- **Residual Connections**: Optional skip connections (ResNet-style)
- **Regularization**: L2 weight decay (categorical choice)

**Layer-Wise Learning Rate**:

```python
class LayerWiseLR(Optimizer):
    """
    Scales LR by 1/√2 for each conv layer (deeper → smaller LR).
    Helps stabilize training in deep networks.
    """
```

**Gradient Monitoring**:

- Tracks `last_grad_global_norm` for vanishing/exploding detection
- Used by symbolic reasoning to trigger normalization fixes

---

### 3. Search Space (`components/search_space.py`)

**Base Dimensions** (always present):

| Parameter           | Type        | Range/Categories         |
| ------------------- | ----------- | ------------------------ |
| `num_neurons`       | Categorical | [8, 16, 32, 64]          |
| `unit_c1/c2`        | Integer     | [1, 4] / [1, 8]          |
| `dr_f`              | Real        | [0.03, 0.5]              |
| `learning_rate`     | Real        | [1e-4, 1e-3]             |
| `batch_size`        | Categorical | [8, 16, 32, 64]          |
| `optimizer`         | Categorical | [Adam, Adamax, SGD, ...] |
| `activation`        | Categorical | [relu, elu, selu, swish] |
| `data_augmentation` | Categorical | [True, False]            |
| `reg_l2`            | Categorical | [True, False]            |
| `skip_connection`   | Categorical | [True, False]            |

**Dynamic Dimensions** (added/removed during tuning):

```python
new_conv_1: Integer(-1, 0)  # -1 = disabled
new_conv_2: Integer(-1, 0)
...
new_fc_1: Integer(-1, 0)
```

**Space Manipulation**:

```python
def add_params(self, params: Dict[str, Any]) -> Space:
    """
    Activates disabled dimensions (low=high=-1 → low=-1, high=value).
    Example: new_conv_3 gets activated with high=64 neurons.
    """

def remove_params(self, params: Dict[str, Any]) -> Space:
    """
    Deactivates dimensions (sets low=high=-1).
    Removes layers deemed unnecessary by symbolic reasoning.
    """
```

**Expansion Strategy** (`expand_space`):

- Merges new dimensions from `next_space` into `base_space`
- Extends categorical choices (e.g., adding optimizer types)
- Widens numeric bounds (min/max over both spaces)
- **Critical**: Preserves dimension order to avoid invalidating BO history

---

## Symbolic Reasoning System

### Architecture

```
Numeric Metrics → Facts → ProbLog Model → Inference → Actions
                    ↓                          ↓
                 Modules               LFI Learning
```

### 1. Neural-Symbolic Bridge (`components/neural_sym_bridge.py`)

**Fact Encoding**:

```prolog
% Training history
l([1.56, 1.22, 1.11, ...]).        % loss
a([0.45, 0.58, 0.67, ...]).        % accuracy
vl([1.78, 1.45, ...]).             % validation loss
va([0.42, 0.55, ...]).             % validation accuracy

% Derived features
int_loss(45.3).                     % integral of loss curve
int_slope(12.1).                    % integral of fitted slope
grad_global_norm(0.0023).           % gradient magnitude

% Hardware constraints (if modules enabled)
flops(125000000).
flops_th(150000000).
hw_latency(0.025).
max_latency(0.033).
```

**Problem Rules** (`symbolic_base/symbolic_analysis.pl`):

```prolog
% Overfitting detection
gap_tr_te_acc :- 
    a(A), va(VA), 
    last(A, LTA), last(VA, ScoreA),
    Res is LTA - ScoreA, 
    abs2(Res, Res1), 
    Res1 > 0.2.

problem(overfitting) :- 
    gap_tr_te_acc, 
    \+ problem(underfitting).

% Gradient issues
vanish_gradient :- 
    grad_global_norm(G), 
    vanish_th(Th), 
    G < Th.

problem(gradient) :- 
    vanish_gradient ; exploding_gradient.
```

**Action Rules** (`symbolic_base/lfi.pl`):

```prolog
% Probabilities learned via LFI
0.70::action(dec_conv_layers, overfitting).
0.50::action(inc_dropout, overfitting).
0.99::action(data_augmentation, underfitting).
0.70::action(add_residual, need_skip).
```

---

### 2. Learning From Interpretations (`components/lfi_integration.py`)

**Experience Accumulation**:

```python
def evidence(self, improve: bool, tuning: List[str], diagnosis: List[str]):
    """
    Creates evidence tuples: (action(tuning, diagnosis), improvement)
    Example: (action(dec_dropout, underfitting), True)
    """
```

**LFI Training**:

```python
def learning(self, improve, tuning, diagnosis, actions):
    """
    Runs ProbLog LFI to update action probabilities.
    Returns: (learned_weights, lfi_problem)
    """
    _, weights, _, _, lfi_problem = lfi.run_lfi(
        PrologString(to_learn), 
        self.experience
    )
```

**Probability Updates**:

- After each iteration, if improvement occurred, increase probabilities of applied actions
- If no improvement, decrease them
- LFI balances exploration (trying new actions) vs. exploitation (proven fixes)

---

### 3. Tuning Rules (`components/tuning_rules_symbolic.py`)

**Example: Adding Residual Connections**:

```python
def add_residual(self) -> None:
    """
    Enables skip connections in the search space.
    Triggered by: problem(need_skip) :- vanish_gradient.
    """
    for i, hp in enumerate(self.space):
        if hp.name == "skip_connection":
            new_categories = [True]
            self.space.dimensions[i] = Categorical(
                new_categories, 
                name='skip_connection'
            )
```

**Example: Architectural Changes**:

```python
def new_conv_block(self) -> None:
    """
    Adds a convolutional section to combat underfitting.
    """
    if self.controller.count_new_cv >= self.controller.max_conv:
        print("Max conv blocks reached")
        return

    self.controller.count_new_cv += 1
    new_p = {f"new_conv_{self.controller.count_new_cv}": 16}
    self.space = self.ss.add_params(new_p)
```

**Repair Orchestration**:

```python
def repair(self, sym_tuning, diagnosis, params, const_space):
    """
    Executes symbolic actions sequentially.
    Maps action names → class methods dynamically.
    """
    for action_name in sym_tuning:
        method = getattr(self, action_name, None)
        if callable(method):
            if action_name in ["decr_lr", "inc_batch_size", ...]:
                method(params)  # Hyperparameter tweaks
            else:
                method()        # Architectural changes
```

---

## Training Pipeline

### 1. Standard Datasets (CIFAR, ImageNet)

```python
# In neural_network.py
history = self.model.fit(
    self.train_data, self.train_labels,
    validation_data=(self.test_data, self.test_labels),
    epochs=self.epochs,
    batch_size=params["batch_size"],
    callbacks=[TensorBoard(), EarlyStopping(patience=20)]
)
```

---

### 2. Early Stopping & Callbacks

```python
es = EarlyStopping(
    monitor="val_accuracy",
    min_delta=0.005,
    patience=20,
    restore_best_weights=True
)
```

**Best Model Persistence**:

- Saved to `{exp_name}/Model/best-model.keras` when score improves
- Dashboard model updated at `dashboard/model/model.keras` after each training

---

## Search Space Management

### Dimension Lifecycle

```
Initial Space
    ↓
┌───────────────────────────────┐
│ Bayesian Optimization Samples │ ← BO history (x0, y0)
└───────────────────────────────┘
    ↓
Training → Diagnosis
    ↓
┌───────────────────────────────┐
│ Symbolic Tuning (add/remove)  │
└───────────────────────────────┘
    ↓
Space Expansion/Contraction
    ↓
┌───────────────────────────────┐
│ Constraint Filtering          │ ← Only if cfg.opt == "filtered"
└───────────────────────────────┘
    ↓
Next BO Sample
```

### Warm-Start vs Cold-Start

**Warm-Start** (space unchanged):

```python
res = gp_minimize(
    obj_fn,
    base_space,
    x0=list(x0),  # Previous evaluations
    y0=list(y0),
    n_calls=1,
    n_random_starts=0  # Use GP surrogate
)
```

**Cold-Start** (space dimensions changed):

```python
if len(next_space.dimensions) != len(const_space.dimensions):
    x0, y0 = [], []  # Reset history!

res = gp_minimize(
    obj_fn,
    base_space,
    n_calls=1,
    n_random_starts=1  # Random sample
)
```

### Constraint Handling

```python
class ConstraintsWrapper:
    def apply_constraints(self, params: List) -> bool:
        """
        Checks if sampled point satisfies current space bounds.
        Used with space_constraint parameter in gp_minimize.
        """
        for i, dim in enumerate(self.space.dimensions):
            val = params[i]
            if isinstance(dim, Categorical):
                if val not in dim.categories:
                    return False
            elif isinstance(dim, (Integer, Real)):
                if not (dim.low <= val <= dim.high):
                    return False
        return True
```


---

## Configuration System

### YAML-Based Configuration (`exp_config.py`)

**File Structure** (`config.yaml`):

```yaml
eval: 300
epochs: 50
mod_list: [flops_module]
dataset: cifar10
name: 2025_01_19_20_30_code_cifar10_filtered_42_300_50
frames: 16
mode: depth
channels: 2
polarity: both
seed: 42
opt: filtered
verbose: 2
quantization: false
early_stop: 30
dataset_path: ./data
cache_dataset: ./cache
created_at: 2025-01-19T20:30:45
```

**Thread-Safe Loading**:

```python
_lock = threading.Lock()
_cached_cfg = None

def load_cfg(force=False) -> DotDict:
    """
    Caches config in memory. Reloads if file modified.
    Uses environment variable EXP_CONFIG for path.
    """
    global _cached_cfg
    with _lock:
        if force or _cached_cfg is None:
            _cached_cfg = _read_yaml(path)
        return _cached_cfg
```

**Dot-Dictionary Access**:

```python
cfg = load_cfg()
print(cfg.dataset)      # "cifar10"
print(cfg.mod_list)     # ["flops_module"]
```

---

## Usage Examples

### 1. Basic Training Run

```bash
python main.py \
    --dataset cifar10 \
    --epochs 50 \
    --eval 100 \
    --opt filtered \
    --name my_experiment \
    --seed 42
```

**What Happens**:

1. Creates `my_experiment/` directory
2. Runs 100 iterations of Bayesian optimization
3. Saves best model to `my_experiment/Model/best-model.keras`
4. Logs training curves to `my_experiment/algorithm_logs/`

---

### 2. Hardware-Aware Tuning

```bash
python main.py \
    --dataset imagenet16120 \
    --mod_list flops_module hardware_module \
    --opt RS_ruled \
    --eval 300 \
    --epochs 100
```

**Module Effects**:

- **flops_module**: Adds constraint `flops(V), flops_th(Th), V > Th → problem(latency)`
- **hardware_module**: Profiles on NVDLA configurations, selects cheapest satisfying latency
- **Symbolic actions**: `dec_conv_layers`, `dec_neurons` triggered when over budget

---


### 3. Analysis (Post-Training)

```bash
# Full analysis (parsing + plots + retraining)
python analyse.py \
    --base-dir ./ \
    --exp-prefix 2025_01_ \
    --plots \
    --train

# Fast aggregation (skip parsing)
python analyse.py \
    --base-dir ./ \
    --aggregate-only
```

**Outputs**:

- `total.csv`: One row per experiment with best accuracy/FLOPs/latency
- `mean.csv`: Grouped statistics (mean ± std) by Tuner × Dataset
- `{exp_name}/search_space_evolution.csv`: Dimension tracking
- `{exp_name}/search_space_pct_changes.png`: Stacked bar chart

---

## Extending the Framework

### 1. Adding a New Loss Module

**Step 1**: Create `modules/loss/my_module.py`

```python
from modules.common_interface import common_interface

class my_module(common_interface):
    facts = ['my_metric', 'my_threshold']
    problems = ['my_problem']
    weight = 0.5  # Contribution to final objective

    def update_state(self, *args):
        self.model = args[0]
        self.my_metric = compute_metric(self.model)

    def obtain_values(self):
        return {'my_metric': self.my_metric, 
                'my_threshold': 100}

    def optimiziation_function(self):
        return -self.my_metric  # Minimize

    def printing_values(self):
        print(f"My Metric: {self.my_metric}")

    def plotting_function(self):
        pass  # Optional visualization

    def log_function(self):
        # Save to file
        with open(f"{cfg.name}/my_module_report.txt", "a") as f:
            f.write(f"{self.my_metric}\n")
```

**Step 2**: Create `modules/loss/my_module.pl`

```prolog
% Define when the problem occurs
high_metric :- my_metric(V), my_threshold(Th), V > Th.
problem(my_problem) :- high_metric.

% Define actions
t(0.80)::action(reduce_layers, my_problem).
t(0.60)::action(increase_dropout, my_problem).
```

**Step 3**: Enable in config

```bash
python main.py --mod_list my_module flops_module
```

---

### 2. Adding a Custom Quantizer

**File**: `quantizer/my_quantizer.py`

```python
from quantizer.quantizer_interface import quantizer_interface

class quantizer_module(quantizer_interface):
    def __init__(self, opt='adam', n_bits=8):
        self.opt = opt
        self.n_bits = n_bits
        self.quantized_model = None

    def quantizer_function(self, model):
        """Implement your quantization logic"""
        # Example: Uniform quantization
        weights = model.get_weights()
        quantized_weights = [
            np.round(w * (2**self.n_bits - 1)) / (2**self.n_bits - 1)
            for w in weights
        ]

        self.quantized_model = tf.keras.models.clone_model(model)
        self.quantized_model.set_weights(quantized_weights)
        return self.quantized_model

    def evaluate_quantized_model(self, x_test, y_test):
        return self.quantized_model.evaluate(x_test, y_test)

    def save_quantized_model(self, path="quantized.keras"):
        self.quantized_model.save(path)
```

**Enable**:

```bash
python main.py --quantization
```

---

### 3. Custom Symbolic Rules

**File**: `symbolic_base/my_rules.pl`

```prolog
% Custom problem detection
oscillating_loss :- 
    l(L), 
    add_to_UpList(L, Up), 
    add_to_DownList(L, Down),
    Ratio is Up / Down,
    Ratio > 0.8, Ratio < 1.2.

problem(instability) :- oscillating_loss.

% Custom actions
0.75::action(increase_batch_size, instability).
0.50::action(reduce_lr, instability).
```

**Include**: The `NeuralSymbolicBridge` automatically loads all `.pl` files in `symbolic_base/`.

---

## Best Practices

### 1. Experiment Organization

```
project/
├── main.py
├── exp_2025_01_20_cifar10_filtered/
│   ├── config.yaml
│   ├── Model/
│   │   └── best-model.keras
│   ├── algorithm_logs/
│   │   ├── acc_report.txt
│   │   ├── hyper-neural.txt
│   │   └── diagnosis_symbolic_logs.txt
│   ├── symbolic/
│   │   ├── lfi.pl          # Learned probabilities
│   │   └── sym_prob.pl     # Current probabilistic model
│   └── results.csv         # Best model summary
```

### 2. Reproducibility

**Always set seed**:

```bash
python main.py --seed 42
```

**Log versions**:

```bash
pip freeze > requirements.txt
git rev-parse HEAD > git_commit.txt
```

---

## Citation

If you use this framework, please cite:

```bibtex
@software{symbolic_dnn_tuner,
  title={Symbolic DNN Tuner: Neuro-Symbolic AutoML for Neural Architecture Search},
  author={Your Name},
  year={2025},
  url={https://github.com/yourusername/symbolic-dnn-tuner}
}
```

---

## License

MIT License - See `LICENSE` file for details.

---

**Documentation Version**: 1.0  
**Last Updated**: January 2025  
**Framework Version**: Compatible with TensorFlow 2.15+, ProbLog 2.2.6+
