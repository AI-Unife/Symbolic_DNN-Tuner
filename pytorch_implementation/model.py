from typing import List
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
from components.model_interface import InsertPosition, LayerSpec, TunerModel, LayerTypes, Params


class TorchModel(TunerModel, nn.Module):

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
    to_type_map = {v: k for k, v in from_type_map.items()}

    def __init__(self, input_shape, params, n_classes, activation_function):
        super(TorchModel, self).__init__()
        
        self.input_shape = input_shape
        self.params = params
        self.n_classes = n_classes
        self.activation = activation_function

        # Layers
        self.modules_list = nn.ModuleList([
            nn.Conv2d(in_channels=input_shape[2], out_channels=params['unit_c1'], kernel_size=3, padding="same"),
            self.activation(),
            nn.Conv2d(in_channels=params['unit_c1'], out_channels=params['unit_c1'], kernel_size=3),
            self.activation(),
            nn.MaxPool2d(kernel_size=2),
            nn.Dropout(params['dr1_2']),

            nn.Conv2d(in_channels=params['unit_c1'], out_channels=params['unit_c2'], kernel_size=3, padding=1),
            self.activation(),
            nn.Conv2d(in_channels=params['unit_c2'], out_channels=params['unit_c2'], kernel_size=3),
            self.activation(),
            nn.MaxPool2d(kernel_size=2),
            nn.Dropout(params['dr1_2'])
        ])

        # Compute the size of the flattened feature map
        with torch.no_grad():
            x = torch.zeros(1, input_shape[2], input_shape[0], input_shape[1])
            for module in self.modules_list:
                x = module(x)
            self.flattened_size = x.view(1, -1).size(1)
        
        self.modules_list.extend([
            nn.Flatten(),
            nn.Linear(self.flattened_size, params['unit_d']),
            self.activation(),
            nn.Dropout(params['dr_f']),
            nn.Linear(params['unit_d'], n_classes)
        ])

        self.create_specs()

    def forward(self, x):
        for module in self.modules_list:
            x = module(x)

        return x
    
    def create_specs(self):
        self.eval()
        dummy_input = torch.zeros([1, self.input_shape[2], self.input_shape[0], self.input_shape[1]])

        def make_hook(module_name):
            def hook_fn(module, input, output):
                self.layers[module_name] = self.to_spec(module_name, module, input, output)

            return hook_fn

        hooks = []
        for name, module in self.modules_list.named_children():
            if not isinstance(module, TorchModel):
                hooks.append(module.register_forward_hook(make_hook(name)))

        self.layers = {}
        with torch.no_grad():
            self(dummy_input)

        for hook in hooks:
            hook.remove()

    def add_layers(self, layers: List[LayerSpec], targets: List[LayerTypes], position: InsertPosition):
        self.eval()
        x = torch.zeros([1, self.input_shape[2], self.input_shape[0], self.input_shape[1]])

        new_module_list = nn.ModuleList([])
        reuse_weights = True

        for name, layer in self.modules_list.named_children():
            if self.to_type(layer.__class__) in targets and position != InsertPosition.After:
                reuse_weights = False
                x, modules = self.fix_shapes(x, layers)
                new_module_list.extend(modules)

                if position == InsertPosition.Replace:
                    continue # Skip current layer (replace it)

            if reuse_weights:
                x = layer(x)
                new_module_list.append(layer)
            else:
                layer_spec = self.layers[name]
                x, modules = self.fix_shapes(x, [layer_spec])
                new_module_list.extend(modules)

            if self.to_type(layer.__class__) in targets and position == InsertPosition.After:
                reuse_weights = False
                x, modules = self.fix_shapes(x, layers)
                new_module_list.extend(modules)

            prev_layer = self.modules_list[-1]

        self.modules_list = new_module_list
        self.create_specs()

    def remove_layers(self, target: LayerSpec, linked_layers: List[LayerTypes], delimiter: bool, first_found: bool):
        self.eval()
        x = torch.zeros([1, self.input_shape[2], self.input_shape[0], self.input_shape[1]])

        removed_names = []
        inside_section = False
        found_section = False
        reuse_weights = True
        new_module_list = nn.ModuleList([])

        for name, layer in self.modules_list.named_children():
            layer_spec = self.layers[name]

            if not inside_section and not found_section:
                # Check if this layer starts a section
                if name == target.name:
                    inside_section = True
                    reuse_weights = False
                    removed_names.append(layer_spec.name)
                    continue

            if inside_section:
                if layer_spec.type not in linked_layers:
                    # End of section
                    inside_section = False
                    found_section = first_found
                    if delimiter:
                        layer_spec = self.layers[name]
                        x, modules = self.fix_shapes(x, [layer_spec])
                        new_module_list.extend(modules)
                    else:
                        # Remove the boundary layer as well
                        removed_names.append(layer_spec.name)
                    continue
                
                # Still inside section
                removed_names.append(layer_spec.name)
                continue

            # Normal case: keep this layer
            if reuse_weights:
                x = layer(x)
                new_module_list.append(layer)
            else:
                layer_spec = self.layers[name]
                x, modules = self.fix_shapes(x, [layer_spec])
                new_module_list.extend(modules)

        print("\n#### removed ####")
        for name in removed_names:
            print(name)

        self.modules_list = new_module_list
        self.create_specs()

    def fix_shapes(self, tensor: torch.Tensor, layer_specs: List[LayerSpec]):
        modules = []
        for layer_spec in copy.deepcopy(layer_specs):
            if layer_spec.type == LayerTypes.Dense:
                layer_spec.set("in_features", tensor.shape[1])
            elif layer_spec.type == LayerTypes.Conv2D:
                layer_spec.set(Params.IN_CHANNELS, tensor.shape[1])
            elif layer_spec.type == LayerTypes.BatchNormalization:
                layer_spec.set("num_features", tensor.shape[1])

                if tensor.dim() == 4: # After Conv2D
                    layer_spec.type = LayerTypes.BatchNormalization2D
                else: # After Linear
                    layer_spec.type = LayerTypes.BatchNormalization1D

            print("Layer type:", layer_spec.type)
            print("Input tensor shape:", tensor.shape)
            print("Params:", layer_spec.params)
        
            module = self.from_spec(layer_spec)
            module.eval()
            modules.append(module)
            tensor = module(tensor)

        return tensor, modules

    def from_type(self, layer_type: LayerTypes):
        return self.from_type_map[layer_type]

    def to_type(self, cls: type[nn.Module]):
        return self.to_type_map[cls]

    def to_spec(self, module_name, module, input, output):
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
                #input_shape=[input[0].shape[0], input[0].shape[2], input[0].shape[3], input[0].shape[1]],
                #output_shape=[output.shape[0], output.shape[2], output.shape[3], output.shape[1]],
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
        elif layer_type in [LayerTypes.ELU, LayerTypes.ReLU, LayerTypes.Sigmoid, LayerTypes.SiLU, LayerTypes.SeLU]:
            return LayerSpec(
                name=module_name, 
                type=layer_type,
                module=module,
                is_activation=True
            )
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

    def from_spec(self, layer_spec):
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
        else:
            return self.from_type(layer_spec.type)()
