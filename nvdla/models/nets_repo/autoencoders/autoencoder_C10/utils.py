import torch
from torch import nn
from torch import softmax
from autoencoder import AutoEncoder

#################
## Test Network
#################
def test(dataloader, model, loss_fn=nn.MSELoss(), verbose=True, complete=True, sample_size=100, device='cpu'):
    model = model.to(device)
    model.eval()
    num_batches = len(dataloader)

    if complete:
        size = len(dataloader.dataset)
        test_loss, correct = 0, 0
        with torch.no_grad():
            for X, y in dataloader:
                X = X.to(device)
                # print(f"Input shape: {X.shape}")
                pred = model(X)
                test_loss += loss_fn(pred, X).item()

        test_loss /= num_batches

        if verbose:
            print(f"Avg loss: {test_loss:>8f} \n")

        return test_loss

############
## Clipper
############
class WeightClipper:
    def __init__(self):
        pass

    def __call__(self, module):
        if hasattr(module, 'weight'):
            w = module.weight.data
            w = w.clamp(-1,1)
            module.weight.data=w

###############
## Train Step
###############
def train_step(dataloader, model, loss_fn, optimizer, device):

    model = model.to(device)
    size=len(dataloader.dataset)

    for batch, (X, y) in enumerate(dataloader):
        X = X.to(device)
        y = X
        #Prediction error
        pred = model(X)
        loss = loss_fn(pred, y)

        #BackPropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        #Clipper
        clipper = WeightClipper()
        model.apply(clipper)

        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

################
## Train Model
################
def train_model(model, dataloader, epochs, loss_fn, optimizer, scheduler, device):
    model = model.to(device)
    model.train()

    # Training
    print("\nFP32 Training Begin...")
    for t in range(epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        train_step(dataloader, model, loss_fn, optimizer, device)
        test(dataloader, model, loss_fn, device)
        scheduler.step()
    print("FP32 Training Complete\n")