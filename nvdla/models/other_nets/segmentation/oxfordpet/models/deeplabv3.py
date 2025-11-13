import torch
import torchvision

def get_deeplabv3(weights_path: str):    
    model = torchvision.models.segmentation.deeplabv3_resnet50(num_classes=3)
    model.load_state_dict(torch.load(weights_path, map_location=torch.device('cpu'))["model"])
    model.eval()

    return model
