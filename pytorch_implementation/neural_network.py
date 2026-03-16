import torch
import json

from time import time

from torch import nn, optim
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from torchinfo import summary

from components.colors import colors
from components.model_interface import InsertPosition, LayerSpec, LayerTypes, Params, TunerModel
from components.neural_network import NeuralNetwork
from pytorch_implementation.model import TorchModel
from components.dataset import TunerDataset


class TorchTunerDataset(Dataset):
    def __init__(self, images: torch.Tensor, labels: torch.Tensor, transform=None):
        self.images = images
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return self.images.shape[0]

    def __getitem__(self, index):
        image = self.images[index]
        if self.transform:
            image = self.transform(image)
        return image, self.labels[index]


class NeuralNetwork (NeuralNetwork):

    def __init__(self, dataset: TunerDataset):
        super().__init__(dataset)

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.checkpoint_format = "pytorch"

        # Framework-specific preprocessing
        if self.dataset.X_train.ndim == 3:
            self.dataset.X_train = self.dataset.X_train[..., None]
            self.dataset.X_test = self.dataset.X_test[..., None]

        self.train_data = self.dataset.X_train
        self.test_data = self.dataset.X_test

        self.train_images = self.to_tensor(self.train_data)
        self.test_images = self.to_tensor(self.test_data)
        self.train_labels = torch.from_numpy(self.dataset.Y_train).long().view(-1)
        self.test_labels = torch.from_numpy(self.dataset.Y_test).long().view(-1)

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

    @staticmethod
    def to_tensor(array):
        if array.ndim == 3:
            array = array[..., None]
        return torch.from_numpy(array).permute(0, 3, 1, 2).contiguous().float()

    def _load_full_model(self, model_path: str):
        try:
            model = torch.load(model_path, map_location=self.device, weights_only=False)
        except TypeError:
            model = torch.load(model_path, map_location=self.device)

        if not isinstance(model, TorchModel):
            raise TypeError(f"Expected TorchModel from checkpoint '{model_path}', got {type(model).__name__}")

        return model

    def from_checkpoint(self, manifest):
        model_path = manifest.get("model_path")
        if not model_path:
            raise ValueError("PyTorch checkpoint missing model_path")

        return self._load_full_model(model_path)

    def from_scratch(self, input_shape, n_classes, params):
        activation_function = self.activation_map[params['activation']]

        return TorchModel(
            input_shape=input_shape, 
            params=params, 
            n_classes=n_classes, 
            activation_function=activation_function
        )

    @staticmethod
    def _l2_penalty(model: TorchModel):
        l2 = None
        for module in model.modules():
            if isinstance(module, nn.Conv2d) and module.weight is not None:
                weight_norm = module.weight.pow(2).sum()
                l2 = weight_norm if l2 is None else l2 + weight_norm

        if l2 is None:
            return torch.tensor(0.0, device=next(model.parameters()).device)

        return l2

    def _evaluate(self, data_loader, criterion, params):
        self.model.eval()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0

        with torch.no_grad():
            for inputs, labels in data_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.model(inputs)
                loss = criterion(outputs, labels)

                if self.rgl:
                    loss = loss + params['reg'] * self._l2_penalty(self.model)

                batch_size = labels.size(0)
                total_loss += loss.item() * batch_size
                total_correct += (outputs.argmax(1) == labels).sum().item()
                total_samples += batch_size

        val_loss = total_loss / total_samples
        val_acc = total_correct / total_samples
        return val_loss, val_acc

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

        # Collect weights to regularize
        if self.rgl:
            self.l2_params = [l.module.weight for l in model.layers.values() if l.type == LayerTypes.Conv2D]

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
        input_shape = self.dataset.X_train.shape[1:]
        
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
                    self.model = self.remove_conv_section(self.model, params)
                # if the flag for the removal of a dense layer is true
                if rem_fc:
                    self.conv = False
                    self.dense = False
                    self.rgl = False
                    self.model = self.remove_fc_section(self.model, params)

        except Exception as e:
            print(colors.FAIL, e, colors.ENDC)

        input_shape = self.dataset.X_train.shape[1:]
        summary(self.model, [1, input_shape[2], input_shape[0], input_shape[1]])
        self.model.to(self.device)

        model_name_id = time()
        self.last_model_id = model_name_id

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

            if isinstance(module, nn.Conv2d):
                lr = params["learning_rate"] * current_mul
                current_mul /= lr_factor
            else:
                lr = params["learning_rate"]

            parameters += [{
                'params': trainable_parameters,
                'lr': lr
            }]

        optimizer = self.optimizer_map[params['optimizer']](parameters)

        # Learning rate adjustment
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, patience=5, min_lr=1e-4)

        # Data augmentation
        if da:
            transform = transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
            ])
        else:
            transform = None

        train_dataset = TorchTunerDataset(self.train_images, self.train_labels, transform=transform)
        train_loader = DataLoader(train_dataset, batch_size=params['batch_size'], shuffle=True, num_workers=0)

        test_dataset = TorchTunerDataset(self.test_images, self.test_labels, transform=None)
        test_loader = DataLoader(test_dataset, batch_size=params['batch_size'], shuffle=False, num_workers=0)

        # Training loop
        history = {'loss': [], 'val_loss': [], 'accuracy': [], 'val_accuracy': []}

        best_val_loss = float('inf')
        best_val_acc = -float('inf')
        patience = 15
        min_delta = 0.005
        counter_loss = 0
        counter_acc = 0

        for epoch in range(self.epochs):
            self.model.train()
            running_loss, correct, total_samples = 0.0, 0, 0
            for inputs, labels in train_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = criterion(outputs, labels)
                # Regularization
                if self.rgl:
                    loss = loss + params['reg'] * self._l2_penalty(self.model)
                loss.backward()
                optimizer.step()
                batch_size = labels.size(0)
                running_loss += loss.item() * batch_size
                correct += (outputs.argmax(1) == labels).sum().item()
                total_samples += batch_size

            train_loss = running_loss / total_samples
            train_acc = correct / total_samples

            # Validation
            val_loss, val_acc = self._evaluate(test_loader, criterion, params)

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

        # Save and reload the trained model for checkpoint continuity across trials
        model_path = f"Model/model-{model_name_id}.pth"
        torch.save(self.model, model_path)
        self.save_manifest({
            "model_path": model_path,
            "params": params,
            "input_shape": list(input_shape),
            "n_classes": self.dataset.n_classes,
        })

        self.model = self._load_full_model(model_path)
        self.model.to(self.device)

        score = self._evaluate(test_loader, criterion, params)
        return [score[0], score[1]], history, self.model
    
    
