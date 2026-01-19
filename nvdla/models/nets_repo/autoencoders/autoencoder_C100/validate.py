import os
import argparse

import torch
from neural_compressor import PostTrainingQuantConfig, quantization
from torch import optim
import torch.nn as nn

from autoencoder import AutoEncoder
from cifar100 import getCIFAR
from utils import test

###################################################

datasetdir = os.environ['TORCH_DATASETDIR']
traindir   = os.environ['TORCH_TRAINDIR']

###################################################

if __name__ == '__main__':
    savefile = os.path.join(traindir, 'fp32_autoencoder_cifar100.pth')

    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default=savefile, help='Path to trained model')
    parser.add_argument('--batchsize', type=int, default=64, help='Batch size for dataloaders')
    parser.add_argument('--device', type=str, default=('cuda' if torch.cuda.is_available() else 'cpu'),
                        help='Device to run evaluation on (cuda or cpu)')
    args = parser.parse_args()

    # Load model
    print(f'-I({__file__}): Loading model...')
    autoencoder = AutoEncoder()
    autoencoder.load_state_dict(torch.load(args.model, map_location=args.device))
    autoencoder.to(args.device)
    print(f'-I({__file__}): Model loaded on {args.device}')

    # Load dataset
    trainloader, testloader = getCIFAR(datasetdir, 32, batchsize=args.batchsize)

    # Debugging: Check a batch
    X, _ = next(iter(testloader))
    print(f"Sample batch shape: {X.shape}")

    # Test training reconstruction loss
    print(f'-I({__file__}): Evaluating Training Loss...')
    train_loss = test(trainloader, autoencoder, loss_fn=nn.MSELoss(), device=args.device)

    # Test testing reconstruction loss
    print(f'-I({__file__}): Evaluating Testing Loss...')
    test_loss = test(testloader, autoencoder, loss_fn=nn.MSELoss(), device=args.device)

    print(f"Training Loss: {train_loss:>8f}")
    print(f"Testing Loss: {test_loss:>8f}")

    # Testing Loss: 0.031371

    model = AutoEncoder().to(device="cpu")
    model.eval()

    def eval_func(model):
        # Use test set for evaluation
        test_loss = test(testloader, model, complete=True, device='cpu', verbose=False)
        return test_loss

    class INCDataLoader:
        def __init__(self, dataloader, device = 'cpu'):
            self.dataloader = dataloader
            self.batch_size = dataloader.batch_size
            self.device = device
        def __iter__(self):
            for X, y in self.dataloader:
                yield X, y
        def __len__(self):
            return len(self.dataloader)


    conf = PostTrainingQuantConfig(approach='static')
    calib_dataloader = INCDataLoader(trainloader)
    q_model = quantization.fit(
        autoencoder,
        conf=conf,
        calib_dataloader=calib_dataloader,
        eval_func=eval_func
    )


    print(f'-I({__file__}): Evaluating Quantized Model...')
    test(testloader, q_model)

    quant_model_path = os.path.join(traindir, 'int8Static_AE_cifar100.pth')
    torch.save(q_model.state_dict(), quant_model_path)


    print(f'-I({__file__}): Saved quantized model to {quant_model_path}')

    # Avg loss: 0.031423