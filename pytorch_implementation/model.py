import re
from typing import List
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
from components.model_interface import InsertPosition, LayerSpec, TunerModel, LayerTypes, Params


class ConvBlock(nn.Module):
    """
    Gestisce un blocco di 'layer_x_block' convoluzioni.
    Se use_residual=True, aggiunge l'input all'output (ResNet style).
    """
    def __init__(self, in_channels, out_channels, num_repeats, activation_fn, use_residual, batch=True):
        super(ConvBlock, self).__init__()
        self.use_residual = use_residual
        self.layers = nn.ModuleList()
        self.activation = activation_fn
        self.batch = batch
        
        # 1. Costruiamo i primi N-1 layer (Conv -> Act -> BN)
        # Nota: In PyTorch è comune l'ordine Conv -> BN -> Act, ma qui replico 
        # la logica Keras Conv -> Act -> BN o simile a seconda delle preferenze.
        # Standard moderno: Conv -> BN -> Act.
        
        current_in = in_channels
        
        # Aggiungiamo (num_repeats - 1) blocchi standard
        for _ in range(num_repeats - 1):
            self.layers.append(nn.Sequential(
                nn.Conv2d(current_in, out_channels, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(out_channels) if batch else nn.Identity(),
                activation_fn
            ))
            current_in = out_channels # Dopo il primo, l'input è out_channels

        # 2. L'ultimo layer del blocco (prima della somma residua)
        self.last_conv = nn.Conv2d(current_in, out_channels, kernel_size=3, padding=1, bias=False)
        self.last_bn = nn.BatchNorm2d(out_channels)

        # 3. Gestione Skip Connection (Shortcut)
        # Se i canali di input sono diversi da quelli di output (es. passaggio da blocco 1 a 2),
        # serve una conv 1x1 per adattare le dimensioni (Projection Shortcut).
        if use_residual:
            self.shortcut = nn.Identity()
            if (in_channels != out_channels):
                self.shortcut = nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
                    nn.BatchNorm2d(out_channels)
                )

    def forward(self, x):
        out = x
        
        # Passaggio attraverso i primi N-1 layer
        for layer in self.layers:
            out = layer(out)
            
        # Ultimo layer convoluzionale (senza attivazione ancora)
        out = self.last_conv(out)
        out = self.last_bn(out) if self.batch else out
        
        # Applicazione Residuale
        if self.use_residual:
            # x + F(x)
            out = out + self.shortcut(x)
        
        # Attivazione finale dopo la somma (ResNet standard)
        out = self.activation(out)
        return out


class TorchModel(TunerModel, nn.Module):
    """
    PyTorch implementation of a TunerModel that extends nn.Module.
    Provides bidirectional mapping between layer types and PyTorch modules,
    along with utilities for converting between layer specifications and actual modules.
    """

    # Maps custom layer types to PyTorch neural network modules
    from_type_map = {
        LayerTypes.Conv2D: nn.Conv2d,
        LayerTypes.MaxPooling2D: nn.MaxPool2d,
        LayerTypes.Dropout: nn.Dropout,
        LayerTypes.Dense: nn.Linear,
        LayerTypes.ELU: nn.ELU,
        LayerTypes.ReLU: nn.ReLU,
        LayerTypes.SiLU: nn.SiLU,
        LayerTypes.SeLU: nn.SELU,
        LayerTypes.Sigmoid: nn.Sigmoid,
        LayerTypes.BatchNormalization1D: nn.BatchNorm1d,
        LayerTypes.BatchNormalization2D: nn.BatchNorm2d,
        LayerTypes.Flatten: nn.Flatten
    }
    # Reverse mapping from PyTorch modules back to custom layer types
    to_type_map = {v: k for k, v in from_type_map.items()}

    def __init__(self, input_shape, params, n_classes, layer_x_block=2, batch=True):
        """
        Initialize the model with the given configuration.
        
        Args:
            input_shape: Tuple of (height, width, channels) for input images
            params: Dictionary containing hyperparameters for layer dimensions
            n_classes: Number of output classes
            activation_function: The activation function to use throughout the model
        """
        super(TorchModel, self).__init__()
        
        self.input_shape = input_shape
        self.params = params
        self.n_classes = n_classes
        self.activation = self._get_activation(params['activation'])
        self.batch = batch
        self.layer_x_block = layer_x_block
        
            # Parsing parametri
        self.use_residual = params.get('skip_connection', False)
        # Istanza dell'attivazione (es. ReLU) da riusare
        act_fn = self._get_activation(params['activation'])
        self.batch = batch
        
        # Calcolo numero canali
        # unit_c * num_neurons
        c1_channels = int(params['unit_c1'] * params['num_neurons'])
        c2_channels = int(params['unit_c2'] * params['num_neurons'])
        
        # --- COSTRUZIONE DELLA RETE ---
        self.features = nn.Sequential()
        in_c = input_shape[0] # (Canali, H, W)
        
        # 1. Blocco C1
        self.features.add_module("block_c1", ConvBlock(
            in_channels=in_c, 
            out_channels=c1_channels, 
            num_repeats=layer_x_block, 
            activation_fn=act_fn, 
            use_residual=self.use_residual,
            batch=self.batch
        ))
        self.features.add_module("pool1", nn.MaxPool2d(2))
        
        # 2. Blocco C2
        self.features.add_module("block_c2", ConvBlock(
            in_channels=c1_channels, 
            out_channels=c2_channels, 
            num_repeats=layer_x_block, 
            activation_fn=act_fn, 
            use_residual=self.use_residual,
            batch=self.batch
        ))
        self.features.add_module("pool2", nn.MaxPool2d(2))
        in_channels = c2_channels
        added_convs = [k for k in params if re.match(r"new_conv_\d+$", k) and params[k] > 0]
        for layer_key in sorted(added_convs, key=lambda s: int(s.split("_")[-1])):  # stable order
            val = params[layer_key]
            out_c = int(val * params['num_neurons'])
            self.features.add_module(f"added_conv_{layer_key}", ConvBlock(
                in_channels=in_channels,
                out_channels=out_c,
                num_repeats=layer_x_block,
                activation_fn=act_fn,
                use_residual=self.use_residual,
                batch=self.batch
            ))
            self.features.add_module(f"pool_{layer_key}", nn.MaxPool2d(2))
            in_channels = out_c  # Update for next layer if any

        # --- CLASSIFICATORE (Fully Connected) ---
        # Calcolo dimensione flatten automatico
        with torch.no_grad():
            dummy = torch.zeros(1, *input_shape)
            out_feat = self.features(dummy)
            
            # SE nel forward fai pooling, devi farlo anche qui!
            if self.batch:
                out_feat = F.adaptive_avg_pool2d(out_feat, (1, 1))
            
            # Ora il flatten darà la dimensione corretta (es. 16 invece di 1024)
            self.flat_dim = out_feat.view(1, -1).size(1)
            
        self.classifier = nn.Sequential()
        current_dim = self.flat_dim
        
        # Aggiunta dinamica layer FC (new_fc_1, new_fc_2, ecc.)
        fc_keys = sorted([k for k in params.keys() if re.match(r'new_fc_\d+', k)], 
                         key=lambda x: int(x.split('_')[-1]))
        
        for i, key in enumerate(fc_keys):
            val = params[key]
            if val > 0:
                out_dim = int(val * params['num_neurons'])
                self.classifier.add_module(f"fc_{i}", nn.Linear(current_dim, out_dim))
                self.classifier.add_module(f"act_{i}", act_fn)
                self.classifier.add_module(f"drop_{i}", nn.Dropout(params['dr_f']))
                current_dim = out_dim
                
        # Layer Output Finale
        self.classifier.add_module("fc_out", nn.Linear(current_dim, self.n_classes))
        
        self.modules_list = nn.ModuleList()
        for module in self.features:
            self.modules_list.append(module)
        for module in self.classifier:
            self.modules_list.append(module)
            
        self.create_specs()

    def _get_activation(self, name):
        name = name.lower()
        if name == 'elu': return nn.ELU(inplace=True)
        if name == 'selu': return nn.SELU(inplace=True)
        return nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.features(x)
        if self.batch:
            # Global Average Pooling: (N, C, H, W) -> (N, C, 1, 1)
            x = F.adaptive_avg_pool2d(x, (1, 1))
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x
    
    def create_specs(self):
        """
        Create LayerSpec objects for each module in the network by hooking into
        the forward pass and capturing input/output information for each layer.
        """
        self.eval()
        # input_shape is (C, H, W), so dummy_input should be (batch, C, H, W)
        dummy_input = torch.zeros([1, self.input_shape[0], self.input_shape[1], self.input_shape[2]])

        def make_hook(module_name):
            """Factory function to create a hook that captures layer specifications"""
            def hook_fn(module, input, output):
                self.layers[module_name] = self.to_spec(module_name, module, input, output)

            return hook_fn

        # Register forward hooks to capture layer information
        # Only register hooks on supported layer types, skip custom modules like ConvBlock
        hooks = []
        for name, module in self.modules_list.named_children():
            if not isinstance(module, (TorchModel, ConvBlock)):
                if module.__class__ in self.to_type_map:
                    hooks.append(module.register_forward_hook(make_hook(name)))

        # Perform forward pass to trigger hooks
        self.layers = {}
        with torch.no_grad():
            self(dummy_input)

        # Clean up hooks
        for hook in hooks:
            hook.remove()


    def from_type(self, layer_type: LayerTypes):
        """
        Get the PyTorch module class for a given layer type.
        
        Args:
            layer_type: Custom layer type enum
            
        Returns:
            Corresponding PyTorch module class
        """
        return self.from_type_map[layer_type]

    def to_type(self, cls: type[nn.Module]):
        """
        Convert a PyTorch module class to its corresponding custom layer type.
        
        Args:
            cls: PyTorch module class
            
        Returns:
            Corresponding custom layer type
        """
        return self.to_type_map[cls]

    def to_spec(self, module_name, module, input, output):
        """
        Convert a PyTorch module into a LayerSpec object based on its type and
        the input/output tensor information.
        
        Args:
            module_name: Name identifier for the module
            module: The PyTorch module to convert
            input: Tuple containing input tensor(s) to the module
            output: Output tensor from the module
            
        Returns:
            LayerSpec object representing the module
        """
        layer_type = self.to_type(module.__class__)
        if layer_type == LayerTypes.Conv2D:
            return LayerSpec(
                name=module_name, 
                type=layer_type,
                module=module,
                params={
                    Params.IN_CHANNELS: input[0].shape[1],
                    Params.IN_HEIGHT: input[0].shape[2],
                    Params.IN_WIDTH: input[0].shape[3],
                    Params.OUT_CHANNELS: output.shape[1],
                    Params.OUT_HEIGHT: output.shape[2],
                    Params.OUT_WIDTH: output.shape[3],
                    Params.KERNEL_SIZE: module.kernel_size,
                    Params.STRIDE: module.stride,
                    Params.PADDING: 'valid' if module.padding[0] == 0 else 'same',
                    Params.BIAS: module.bias is not None
                }
            )
        elif layer_type == LayerTypes.Dense:
            return LayerSpec(
                name=module_name, 
                type=layer_type,
                module=module,
                params={
                    Params.IN_FEATURES: input[0].shape[1],
                    Params.OUT_FEATURES: output.shape[1],
                }
            )
        elif layer_type == LayerTypes.MaxPooling2D:
            return LayerSpec(
                name=module_name, 
                type=layer_type,
                module=module,
                params={
                    Params.KERNEL_SIZE: module.kernel_size,
                    Params.STRIDE: module.stride
                }
            )
        elif layer_type == LayerTypes.Dropout:
            return LayerSpec(
                name=module_name, 
                type=layer_type,
                module=module,
                params={
                    Params.DROPOUT_RATE: module.p
                }
            )
        # Activation function layers
        elif layer_type in [LayerTypes.ELU, LayerTypes.ReLU, LayerTypes.Sigmoid, LayerTypes.SiLU, LayerTypes.SeLU]:
            return LayerSpec(
                name=module_name, 
                type=LayerTypes.Activation,
                module=module,
                is_activation=True,
                params={
                    "activation_type": layer_type
                }
            )
        # Batch normalization layers
        elif layer_type in [LayerTypes.BatchNormalization1D, LayerTypes.BatchNormalization2D]:
            return LayerSpec(
                name=module_name,
                type=layer_type,
                module=module,
                params={
                    Params.NUM_FEATURES: module.num_features
                }
            )
        elif layer_type == LayerTypes.Flatten:
            return LayerSpec(
                name=module_name,
                type=layer_type,
                module=module,
                params={
                    "start_dim": module.start_dim,
                    "end_dim": module.end_dim
                }
            )
        else:
            raise Exception("Missing LayerSpec for layer of type " + module.__class__.__name__)

    def from_spec(self, layer_spec: LayerSpec):
        """
        Create a PyTorch module from a LayerSpec object.
        
        Args:
            layer_spec: Layer specification containing type and parameters
            
        Returns:
            Initialized PyTorch module
        """
        if layer_spec.type == LayerTypes.Conv2D:
            return nn.Conv2d(
                in_channels=layer_spec.get(Params.IN_CHANNELS),
                out_channels=layer_spec.get(Params.OUT_CHANNELS),
                kernel_size=layer_spec.get(Params.KERNEL_SIZE),
                padding=layer_spec.get(Params.PADDING),
            )
        elif layer_spec.type == LayerTypes.Dense:
            return nn.Linear(
                in_features=layer_spec.get('in_features'),
                out_features=layer_spec.get('out_features')
            )
        elif layer_spec.type == LayerTypes.MaxPooling2D:
            return nn.MaxPool2d(
                kernel_size=layer_spec.get(Params.KERNEL_SIZE),
                stride=layer_spec.get(Params.STRIDE)
            )
        elif layer_spec.type == LayerTypes.Dropout:
            return nn.Dropout(
                p=layer_spec.get(Params.DROPOUT_RATE)
            )
        elif layer_spec.type in [LayerTypes.BatchNormalization1D, LayerTypes.BatchNormalization2D]:
            return self.from_type(layer_spec.type)(
                num_features=layer_spec.get(Params.NUM_FEATURES)
            )
        elif layer_spec.type == LayerTypes.Flatten:
            return nn.Flatten(
                start_dim=layer_spec.get("start_dim"),
                end_dim=layer_spec.get("end_dim")
            )
        elif layer_spec.type == LayerTypes.Activation:
            return self.from_type(layer_spec.get("activation_type"))()
        else:
            return self.from_type(layer_spec.type)()
