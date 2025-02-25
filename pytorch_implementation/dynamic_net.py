import time

from torch import nn
from components.dynamic_net import dynamic_net

class PytorchDynamicNet (dynamic_net):

    def any_batch(self, model):
        """
        Check if at least one BatchNorm operation is in the PyTorch model.
        :param model: PyTorch model in which to detect BatchNorm layers.
        :return: Boolean indicating if there's at least one BatchNorm layer.
        """
        return any(isinstance(layer, nn.BatchNorm2d) for layer in model.modules())

    def count_layer_type(self, model, layer_type):
        """
        Count how many layers of a certain type are in the PyTorch model.
        :param model: PyTorch model in which to count the number of 'layer_type' layers.
        :param layer_type: The layer type (e.g., nn.Conv2d, nn.Linear) to count.
        :return: Number of layers of the specified type.
        """
        return sum(1 for layer in model.modules() if isinstance(layer, layer_type))

    def insert_section(self, model, n_section, new_section, position, target):
        """
        Insert a new section into an `nn.Sequential` model.
        
        :param model: The original nn.Sequential model
        :param n_section: Number of times to replicate the new_section
        :param new_section: List of layers to insert
        :param position: Where to insert the new section ('before', 'after', or 'replace')
        :param target: The name or class of the layer to target
        :return: Modified nn.Sequential model
        """
        if not self.all_layers(new_section):
            print("\n#### New section contains elements that are not layers ####\n")
            return model

        # Get all layers from the model
        layers_list = list(model.named_children())

        # Prepare the new section
        extended_new_section = []
        for _ in range(n_section):
            extended_new_section.extend(new_section)
        named_new_section = self.add_names(extended_new_section)

        # Create the new layer order
        new_layers = []
        for name, layer in layers_list:
            layer_class = layer.__class__.__name__

            if (layer_class in target or name in target):
                if position == "before":
                    new_layers.extend(named_new_section)
                elif position == "replace":
                    new_layers.extend(named_new_section)
                    continue  # Skip the current layer (replace it)

            new_layers.append((name, layer))
            if (layer_class in target or name in target) and position == "after":
                new_layers.extend(named_new_section)

        # Build the new Sequential model
        return nn.Sequential(dict(new_layers))
    
    # TODO: This differs from the tensorflow implementation as it doesn't append the new name to the existing name
    # This is because in Pytorch, layers are not named, unless using nn.Sequential but then custom naming must be given
    # when building the new section (eg. in neural_network.insert_conv_section())
    def add_names(self, layers):
        """
        Add unique names to layers by wrapping them in a dictionary with unique keys.
        """
        named_layers = []
        for i, layer in enumerate(layers):
            named_layers.append((f"{time.time()}_{i}", layer))
        return named_layers

    def model_from_dict(self, model, model_dict):
        """
        Build a PyTorch model based on a dict containing layer configurations.
        :param model: Original PyTorch model.
        :param model_dict: Dictionary with layer configurations.
        :return: New PyTorch model.
        """
        new_layers = nn.ModuleDict()
        input_shape = None

        # Layer name list from the original model
        name_list = [name for name, _ in model.named_modules()]

        for layer_key, layer_config in model_dict.items():
            reused_weights, layer_type, layer_params = layer_config

            if layer_type == "Input":
                input_shape = layer_params.get("input_shape")
                continue  # Input layers are implicit in PyTorch
            
            if reused_weights and layer_key in name_list:
                # Reuse the existing layer with pretrained weights
                new_layers[layer_key] = getattr(model, layer_key)
            else:
                # Instantiate a new layer using the layer configuration
                layer = self.instantiate_layer(layer_type, layer_params)
                if layer:
                    new_layers[layer_key] = layer

        # Create the final sequential model
        new_model = nn.Sequential(new_layers)
        return new_model
    
    def get_last_section(self, model, type_class):
        """
        Method to find the index of the first layer in the last section of a specific type.
        :param model: PyTorch nn.Sequential model to search
        :param type_class: Class of the layer type to search for (e.g., torch.nn.Conv2d)
        :return: Index of the layer where the last section begins
        """
        # Get the layers of the model in reverse order
        layers = list(model.children())[::-1]

        # Variables to track the section
        last_section_start = None
        section_found = False

        # Iterate over layers in reverse order
        for i, layer in enumerate(layers):
            # Check if the current layer is of the desired type
            if isinstance(layer, type_class) and not section_found:
                section_found = True
                last_section_start = len(layers) - 1 - i  # Convert reverse index to original index
            elif section_found and not isinstance(layer, type_class):
                # If we find a layer that isn't of the target type, stop searching
                break

        return last_section_start
    
    # TODO: find out how reusing weights works in pytorch
    # I think right now i'm reusing weights for all layers
    def remove_section(self, model, target, linked_layers, delimiter, first_found):
        """
        Method used to remove a section of layers in a PyTorch nn.Sequential model.
        :param model: PyTorch Sequential model
        :param target: Layer type or name to start removing
        :param linked_layers: List of layer types or names connected to the target
        :param delimiter: Boolean to decide if the process is delimited by a specific condition
        :param first_found: Boolean to stop after finding the first matching section
        :return: New model without the linked_layers
        """

        # Boolean to track reuse of old weights
        reused_weights = True

        # Flags for section removal
        n_section, n_found = False, False

        # Initialize dictionary to store layers after removal
        remaining_layers = []

        # Track names of removed layers for debugging
        removed_name = ""

        # Iterate over the layers in the Sequential model
        for name, layer in model.named_children():
            layer_class = layer.__class__.__name__

            # If not in a section and target layer is found
            if not n_section and not n_found and (target in layer_class or target in name):
                n_section = True
                removed_name += name + '\n'
                reused_weights = False  # Layers after this cannot reuse old weights

            elif n_section:
                # If the current layer is not linked, end the section
                if layer_class not in linked_layers and name not in linked_layers:
                    n_section = False
                    n_found = first_found

                    if delimiter:
                        remaining_layers.append((name, layer))
                    else:
                        removed_name += name + '\n'
                else:
                    removed_name += name + '\n'
            else:
                # Add the current layer to the remaining layers
                remaining_layers.append((name, layer))

        print(f"\n#### removed ####\n{removed_name}")

        # Rebuild the Sequential model with remaining layers
        new_model = nn.Sequential()
        for name, layer in remaining_layers:
            new_model.add_module(name, layer)

        return new_model
    
    def build_model(self, model, model_dict):
        pass

    def all_layers(self, layer_list):
        """
        Method to check if all elements of a list are PyTorch layers.
        
        :param layer_list: List of input layers
        :return: Boolean indicating if all elements are PyTorch layers
        """
        return all(isinstance(layer, nn.Module) for layer in layer_list)