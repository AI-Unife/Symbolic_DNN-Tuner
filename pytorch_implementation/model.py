import torch
import torch.nn as nn
import torch.nn.functional as F
 
 
class Model(torch.nn.Module):
 
    def __init__(self, input_shape, n_classes, params):
        super(Model, self).__init__()

        self.params = params
 
        # Define layers
        self.conv1 = nn.Conv2d(in_channels=input_shape[0], out_channels=params['unit_c1'], kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=params['unit_c1'], out_channels=params['unit_c1'], kernel_size=3)
        self.conv3 = nn.Conv2d(in_channels=params['unit_c1'], out_channels=params['unit_c2'], kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(in_channels=params['unit_c2'], out_channels=params['unit_c2'], kernel_size=3)
        
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout1 = nn.Dropout(params['dr1_2'])
        self.dropout2 = nn.Dropout(params['dr1_2'])
        self.flatten = nn.Flatten()
        self.activation = getattr(F, params['activation'])
        
        # TODO: self.fc1 is initialized during the first forward pass to calculate input size dynamically
        # input size can also be calculated when defining the network (more efficient) in case my input shape is fixed
        self.fc1 = None
        self.fc2 = nn.Linear(params['unit_d'], n_classes)
        self.dropout_fc = nn.Dropout(params['dr_f'])

    def forward(self, x):
        x = self.activation(self.conv1(x))
        x = self.activation(self.conv2(x))
        x = self.pool(x)
        x = self.dropout1(x)
        
        x = self.activation(self.conv3(x))
        x = self.activation(self.conv4(x))
        x = self.pool(x)
        x = self.dropout2(x)
        
        x = self.flatten(x)
        
        if self.fc1 is None:
            self.fc1 = nn.Linear(x.size(1), self.params['unit_d']).to(x.device)
        
        x = self.activation(self.fc1(x))
        x = self.dropout_fc(x)
        x = self.fc2(x)
        x = F.softmax(x, dim=1)
        return x