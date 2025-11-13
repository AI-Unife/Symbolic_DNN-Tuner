import os
import torch

def get_ssd(weights_dir):
    torch.hub.set_dir(weights_dir)
    model = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_ssd')
    utils = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_ssd_processing_utils')
    model.eval()
    return model