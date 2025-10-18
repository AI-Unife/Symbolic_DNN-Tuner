import torch

import time

from torch import nn, optim
from torchvision import transforms
import torchvision
from torch.utils.data import DataLoader, TensorDataset
from torchinfo import summary

from components.colors import colors
from components.model_interface import InsertPosition, LayerSpec, LayerTypes, Params, TunerModel
from components.neural_network import neural_network
from pytorch_implementation.model import TorchModel

class NeuralNetwork (neural_network):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.activation_map = {
            "relu": nn.ReLU,
            "elu": nn.ELU,
            "selu": nn.SELU,
            "swish": nn.SiLU
        }

        # Map optimizers
        self.optimizer_map = {
            "Adam": optim.Adam,
            "Adamax": optim.Adamax,
            "Adagrad": optim.Adagrad,
            "Adadelta": optim.Adadelta
        }

    def from_checkpoint(self, checkpoint):
        params = checkpoint["params"]
        input_shape = tuple(checkpoint["input_shape"])
        n_classes = checkpoint("n_classes")

        model = self.from_scratch(input_shape, n_classes, params)

        if "state_dict" in checkpoint:
            model.load_state_dict(torch.load(checkpoint["state_dict"]))

        return model

    def from_scratch(self, input_shape, n_classes, params):
        activation_function = self.activation_map[params['activation']]

        return TorchModel(
            input_shape=input_shape, 
            params=params, 
            n_classes=n_classes, 
            activation_function=activation_function
        )

    def insert_batch(self, model: TunerModel, params):
        """
        Inserts BatchNormalization operations into the PyTorch model.
        :param model: PyTorch model to modify.
        :param params: Parameters for batch normalization and regularization.
        :return: Modified model with BatchNormalization and regularization.
        """
        # If BatchNormalization is already in the model, return it
        if (self.dnet.count_layer_type(model, LayerTypes.BatchNormalization1D) > 1 or
                self.dnet.count_layer_type(model, LayerTypes.BatchNormalization2D) > 1):
            return model

        # TODO: PyTorch does not directly support attaching regularizers to layers
        # In Pytorch i need to manually compute L2 regularization
        # Add regularization to each convolutional layer
        # for layer in model.modules():
        #     if self.rgl and isinstance(layer, nn.Conv2d):
        #         layer.weight_regularizer = nn.L1Loss(params['reg'])

        # Collect activations to which BatchNormalization should be added
        activation_list = [layer.type for layer in model.layers.values() if layer.is_activation and layer.type != LayerTypes.Softmax]

        new_section = [
            LayerSpec(
                type=LayerTypes.BatchNormalization
            )
        ]

        # Apply BatchNormalization to all collected activations
        return self.dnet.insert_section(model, 1, new_section, InsertPosition.After, activation_list)

    def training(self, params, new, new_fc, new_conv, rem_conv, rem_fc, da, space):

        self.model = self.build_network(params, new)
        input_shape = self.train_data.shape[1:]
        
        # summary(self.model, [1, input_shape[2], input_shape[0], input_shape[1]])
        # print(self.model)

        # self.model = self.insert_batch(self.model, params)
        # params['new_fc'] = 512
        # self.model = self.insert_fc_section(self.model, params, 1)
        # self.model = self.insert_conv_section(self.model, params, 1)
        # self.model = self.remove_conv_section(self.model)
        # self.model = self.remove_fc_section(self.model)
        
        # summary(self.model, [1, input_shape[2], input_shape[0], input_shape[1]])
        # print(self.model)
        # exit()

        try:
            # try adding or removing a layer in the neural network based on the anomalies diagnosis
            if new or new_fc or new_conv or rem_conv:
                # if the flag for the addition of a dense layer is true
                if new_fc:
                    if new_fc[0]:
                        self.dense = True
                        self.model = self.insert_fc_section(self.model, params, new_fc[1])
                # if the flag for the addition of regularization is true
                if new:
                    self.rgl = True
                    self.dense = False
                    self.model = self.insert_batch(self.model, params)
                # if the flag for the addition of a convolutional layer
                if new_conv:
                    if new_conv[0]:
                        self.conv = True
                        self.dense = False
                        self.rgl = False
                        self.model = self.insert_conv_section(self.model, params, new_conv[1])
                # if the flag for the removal of a convolutional layer is true
                if rem_conv:
                    self.conv = False
                    self.dense = False
                    self.rgl = False
                    self.model = self.remove_conv_section(self.model)
                # if the flag for the removal of a dense layer is true
                if rem_fc:
                    self.conv = False
                    self.dense = False
                    self.rgl = False
                    self.model = self.remove_fc_section(self.model)

        except Exception as e:
            print(colors.FAIL, e, colors.ENDC)

        input_shape = self.train_data.shape[1:]
        summary(self.model, [1, input_shape[2], input_shape[0], input_shape[1]])

        # Save model architecture
        model_name_id = str(int(time.time()))
        model_path = f"Model/model-{model_name_id}.pth"
        torch.save(self.model.state_dict(), model_path)

        # Define optimizer and loss function
        criterion = nn.CrossEntropyLoss()

        # Layer wise learning rate
        parameters = []
        current_mul = 1
        lr_factor = 1.414213

        for module in self.model.modules_list:
            trainable_parameters = [p for n, p in module.named_parameters() if p.requires_grad]

            if not len(trainable_parameters):
                continue

            if module.type == LayerTypes.Conv2D:
                lr = params["learning_rate"] * current_mul
                current_mul /= lr_factor
            else:
                lr = params["learning_rate"]

            parameters += [{
                'params': trainable_parameters,
                'lr': lr
            }]

        # optimizer = getattr(optim, params['optimizer'])(self.model.parameters(), lr=params['learning_rate'])
        optimizer = getattr(optim, params['optimizer'])(parameters)

        # Learning rate adjustment
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, patience=5, min_lr=1e-4)

        # Data augmentation
        if da:
            transform = transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
                transforms.ToTensor(),
            ])
        else:
            transform = transforms.Compose([transforms.ToTensor()])

        train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=params['batch_size'],
                                          shuffle=True, num_workers=0)

        test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                        download=True, transform=transform)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=params['batch_size'],
                                          shuffle=False, num_workers=0)

        # Training loop
        history = {'loss': [], 'val_loss': [], 'accuracy': [], 'val_accuracy': []}
        best_loss = float('inf')

        best_val_loss = float('inf')
        best_val_acc = -float('inf')
        patience = 15
        min_delta = 0.005
        counter_loss = 0
        counter_acc = 0

        for epoch in range(self.epochs):
            self.model.train()
            running_loss, correct = 0.0, 0
            for inputs, labels in train_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
                correct += (outputs.argmax(1) == labels).sum().item()

            train_loss = running_loss / len(train_loader)
            train_acc = correct / len(self.train_data)

            # Validation
            self.model.eval()
            val_loss, val_correct = 0.0, 0
            with torch.no_grad():
                for inputs, labels in test_loader:
                    inputs, labels = inputs.to(self.device), labels.to(self.device)
                    outputs = self.model(inputs)
                    loss = criterion(outputs, labels)
                    val_loss += loss.item()
                    val_correct += (outputs.argmax(1) == labels).sum().item()

            val_loss /= len(test_loader)
            val_acc = val_correct / len(self.test_data)

            # Update history
            history['loss'].append(train_loss)
            history['val_loss'].append(val_loss)
            history['accuracy'].append(train_acc)
            history['val_accuracy'].append(val_acc)

            print(f"Epoch {epoch+1}: loss={train_loss:.4f}, val_loss={val_loss:.4f}, acc={train_acc:.4f}, val_acc={val_acc:.4f}")

            scheduler.step(val_loss)

            # Early stopping on val_loss (mode='min')
            if val_loss < best_val_loss - min_delta:
                best_val_loss = val_loss
                counter_loss = 0
            else:
                counter_loss += 1

            # Early stopping on val_acc (mode='max')
            if val_acc > best_val_acc + min_delta:
                best_val_acc = val_acc
                counter_acc = 0
            else:
                counter_acc += 1

            # Stop if either condition reaches patience
            if counter_loss >= patience or counter_acc >= patience:
                print("Early stopping triggered.")
                break

        # Load best model
        self.model.load_state_dict(torch.load(model_path))

        # TODO: Maybe val_loss
        return [best_loss, train_acc], history, self.model
    
    