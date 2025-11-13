import torch
import torch.nn as nn
import torchvision

def get_mobilenetv2_model(n_classes, weights_path):
    model = torchvision.models.mobilenet_v2()

    model.classifier[1] = nn.Linear(1280, n_classes, bias=True)
    for param in model.classifier[1].parameters():
        param.requires_grad = True

    weights = torch.load(weights_path, map_location=torch.device('cpu'), weights_only=False)['model']
    model.load_state_dict(weights)
    model.eval()

    return model
