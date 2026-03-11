import torch
import torch.nn as nn
from torch.utils.flop_counter import FlopCounterMode

from components.backend_interface import BackendInterface

class ModuleBackend(BackendInterface):
    def build_lenet(self):
        """Build LeNet model using Sequential"""
        model = nn.Sequential(
            nn.Conv2d(1, 6, kernel_size=5, padding=2),
            nn.Tanh(),
            nn.AvgPool2d(2),
            nn.Sigmoid(),
            nn.Conv2d(6, 16, kernel_size=5),
            nn.Tanh(),
            nn.AvgPool2d(2),
            nn.Sigmoid(),
            nn.Flatten(),
            nn.Linear(16 * 5 * 5, 120),
            nn.Tanh(),
            nn.Linear(120, 84),
            nn.Tanh(),
            nn.Linear(84, 10)
        )
        return model

    def get_flops(self, model, input_shapes):
        istrain = model.training
        model.eval()
        
        input_shapes = (1, input_shapes[2], input_shapes[0], input_shapes[1])
        input_shapes = input_shapes if isinstance(input_shapes, torch.Tensor) else torch.randn(input_shapes)

        flop_counter = FlopCounterMode(mods=model, display=False, depth=None)
        with flop_counter:
            model(input_shapes).sum().backward() # Forward and backward
            # model(input_shapes) # Forward only
            total_flops =  flop_counter.get_total_flops()

        if istrain:
            model.train()

        return total_flops
