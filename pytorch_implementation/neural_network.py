import torch

from torch import nn

from components.neural_network import neural_network
from pytorch_implementation.model import Model

class PytorchNeuralNetwork (neural_network):

    def from_checkpoint(checkpoint):
        params = checkpoint["params"]
        input_shape = tuple(checkpoint["input_shape"])
        n_classes = checkpoint("n_classes")

        model = Model(input_shape, n_classes, params)

        if "state_dict" in checkpoint:
            model.load_state_dict(torch.load(checkpoint["state_dict"]))

        return model

    def from_scratch(input_shape, n_classes, params):
        return Model(input_shape, n_classes, params)

    def insert_conv_section(self, model, params, n_conv):
        """
        Inserts a convolutional section into a PyTorch model.
        :param model: PyTorch model to modify.
        :param params: Dictionary of parameters for the new section.
        :param n_conv: Position at which to insert the new convolutional section.
        :return: Modified model with the new conv section added.
        """
        # Count the current number of convolutional layers in the model
        current_conv_count = self.dnet.count_layer_type(model, nn.Conv2d)

        # Initialize the counter of new convolutions and the list of new layers
        new_conv_count = 0
        new_section = []

        # Cycle to add at most two convolutions in the new conv section
        for i in range(2):
            # Increase the counter to add a new convolutional layer
            new_conv_count += 1

            # If the sum of current and new convolutions exceeds the limit, stop adding layers
            if (new_conv_count + current_conv_count) > self.tot_conv:
                break

            # Add a convolutional layer and its activation
            new_section.append(nn.Conv2d(params['in_channels'], params['unit_c2'], kernel_size=3, padding=1))
            activation = getattr(nn, params['activation'], None)
            if activation is not None:
                new_section.append(activation())

            # If batch normalization exists in the model, add it to the new convolutional section
            if self.dnet.any_batch(model):
                new_section.append(nn.BatchNorm2d(params['unit_c2']))

        # If the new section is empty, return the original model
        if not new_section:
            return model

        # Add max pooling and dropout to the convolutional section
        new_section.append(nn.MaxPool2d(kernel_size=2, stride=2))
        new_section.append(nn.Dropout(params['dr_f']))

        # Insert the new section before the Flatten layer
        return self.dnet.insert_section(model, n_conv, new_section, 'before', "Flatten")

    def insert_batch(self, model, params):
        """
        Inserts BatchNormalization operations into the PyTorch model.
        :param model: PyTorch model to modify.
        :param params: Parameters for batch normalization and regularization.
        :return: Modified model with BatchNormalization and regularization.
        """
        # If BatchNormalization is already in the model, return it
        if self.dnet.any_batch(model):
            return model

        # Add regularization to each convolutional layer
        for layer in model.modules():
            if self.rgl and isinstance(layer, nn.Conv2d):
                # TODO: PyTorch does not directly support attaching regularizers (like L1 or L2) to layers
                # In Pytorch i think i need to pass weight_decay to the optimizer.
                layer.weight_regularizer = nn.L1Loss(params['reg'])

        # Collect activations to which BatchNormalization should be added
        activation_list = []
        for name, layer in model.named_modules():
            if isinstance(layer, nn.ReLU) and not hasattr(layer, 'is_output'):
                activation_list.append(name)

        # Apply BatchNormalization to all collected activations
        return self.dnet.insert_section(model, 1, [nn.BatchNorm2d], 'after', activation_list)

    def insert_fc_section(self, model, params, n_fc):
        """
        Method for inserting a fully connected section into a PyTorch model.
        
        :param model: The PyTorch model to modify.
        :param params: Dictionary with the parameters for the new section, including:
                       - 'new_fc': Number of units in the new fully connected layer.
                       - 'activation': Activation function as a string (e.g., 'relu', 'sigmoid').
                       - 'dr_f': Dropout rate (e.g., 0.5 for 50% dropout).
        :param n_fc: Position index where the new section should be inserted.
        :return: Modified model with the new fully connected section.
        """
        # Check if the number of dense layers in the model exceeds or equals the maximum allowed
        if self.dnet.count_layer_type(model, nn.Linear) >= self.tot_fc:
            return model
        
        # Build the new fully connected section
        new_section = [nn.Linear(params['new_fc'], params['new_fc'])]
        
        # TODO: this if/elif block can be rewritten in a single line using a dictionary
        # Add activation function
        if params['activation'] == 'relu':
            new_section.append(nn.ReLU())
        elif params['activation'] == 'sigmoid':
            new_section.append(nn.Sigmoid())
        elif params['activation'] == 'tanh':
            new_section.append(nn.Tanh())
        else:
            raise ValueError(f"Unsupported activation: {params['activation']}")
        
        # If batch normalization is already in the model, add it to the new section
        if self.dnet.any_batch(model):
            new_section.append(nn.BatchNorm1d(params['new_fc']))
        
        # Add dropout layer
        new_section.append(nn.Dropout(params['dr_f']))
        
        # Insert the new section at the specified position
        return self.dnet.insert_section(model, n_fc, new_section, 'after', nn.Flatten)

    def remove_conv_section(self, model):
        """
        Method to remove a convolutional section from a PyTorch model.
        :param model: PyTorch model from which to remove the convolutional section
        :return: model without the convolutional section
        """
        # Check the number of Conv2D layers
        current_conv_count = self.dnet.count_layer_type(model, torch.nn.Conv2d)
        if current_conv_count <= 1:
            return model

        # Get the name (or index) of the first layer of the last convolutional section
        last_conv_start = self.dnet.get_last_section(model, torch.nn.Conv2d)
        
        # Define the layers associated with the convolutional section
        linked_section = [torch.nn.Conv2d, torch.nn.ReLU, torch.nn.BatchNorm2d]

        # If the number of convolutions is odd, include additional layers to remove
        if (current_conv_count % 2) == 1:
            linked_section += [torch.nn.MaxPool2d, torch.nn.Dropout]

        # Use the helper function to remove the section
        return self.dnet.remove_section(model, last_conv_start, linked_section, True, True)
    
    def remove_fc_section(self, model):
        """
        Method used for removing a dense (fully connected) section.
        :param model: model from which to remove the dense section
        :return: model without the dense section
        """

        # If the number of dense (Linear) layers is less than or equal to 2,
        # specifically a dense layer after the flatten and the output,
        # don't remove any dense layer and return the model
        if self.dnet.count_layer_type(model, nn.Linear) <= 2:
            return model

        # Remove the first dense section in the model and all associated layers in linked_section
        linked_section = [nn.ReLU, nn.BatchNorm1d, nn.Dropout]
        return self.dnet.remove_section(model, nn.Linear, linked_section, True, True)