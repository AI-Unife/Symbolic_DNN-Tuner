import torchvision
import torch.nn as nn

# from torchinfo import summary

def get_mobilenetv2_model(n_classes, pretrained_weights=True, return_transforms=True):
    weights = torchvision.models.MobileNet_V2_Weights.IMAGENET1K_V1
    if pretrained_weights:
        model = torchvision.models.mobilenet_v2(weights=weights)
    else:
        model = torchvision.models.mobilenet_v2()


    model.classifier[1] = nn.Linear(1280, n_classes, bias=True)
    for param in model.classifier[1].parameters():
        param.requires_grad = True

    model.eval()
    # summary(model, row_settings=('var_names',), input_size=(1,3,256,256), col_names=('input_size', 'output_size'), depth=5)

    if return_transforms:
        return model, weights.transforms()
    else:
        return model
