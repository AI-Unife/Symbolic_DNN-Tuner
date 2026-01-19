import time
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import sys
from vision_models_pytorch.models.vgg16_ssd.model import SSD300, MultiBoxLoss
from vision_models_pytorch.models.vgg16_ssd.datasets.pascal_voc import PascalVOCDataset
from vision_models_pytorch.utils.training import (
    adjust_learning_rate,
    save_checkpoint,
    clip_gradient,
)
from vision_models_pytorch.utils.metrics import AverageMeter

from tqdm import tqdm

# Data parameters
data_folder = "./"  # folder with data files
keep_difficult = True  # use objects considered difficult to detect?

# Model parameters
# Not too many here since the SSD300 has a very specific structure
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

cudnn.benchmark = True


def train_ssd(
    # Learning parameters
    train_loader=None,
    checkpoint_output_path="checkpoint_ssd300.pth.tar",
    restarting_checkpoint_path=None,  # path to model checkpoint, None if none
    iterations=24000,  # number of iterations to train
    print_freq=200,  # print training status every __ batches
    lr=1e-3,  # learning rate
    n_classes=10,
    decay_lr_at=[16000, 30000],  # decay learning rate after these many iterations
    decay_lr_to=0.1,  # decay learning rate to this fraction of the existing learning rate
    momentum=0.9,  # momentum
    weight_decay=5e-4,  # weight decay
    grad_clip=None,  # clip if gradients are exploding, which may happen at larger batch sizes (sometimes at 32) - you will recognize it by a sorting error in the MuliBox loss calculation
):
    """
    Perform training of SSD model.
    The default values are those used for training using the VOC dataset proposed in the original repo

    Params
    ---
    -
    - ``restarting_checkpoint_path``: Path of the checkpoint used for resuming training. If None training resumes from scratch
    - ``batch_size``: Size of batch training
    - ``iterations``: Number of iteration of training (Number of batches)
    - ``loader_workers``: Number of workers for loading data in the DataLoader
    - ``print_freq``:  Print training status every ``print_freq`` batches
    - ``lr``: Learning rate of the optimizer
    - ``decay_lr_at``: Decay Learning rate after these many iterations, multiplying it by ``decay_lr_to``
    - ``decay_lr_to``: decay learning rate to this fraction of the existing learning rate
    - ``momentum``: Momentum of the optimizer
    - ``weight_decay``: Weight decay for regularization of the model
    - ``grad_clip``: clip if gradients are exploding, which may happen at larger batch sizes (sometimes at 32)
    """

    if train_loader is None:
        train_dataset = PascalVOCDataset(
            data_folder, split="train", keep_difficult=keep_difficult
        )
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=16,
            shuffle=True,
            collate_fn=train_dataset.collate_fn,
            num_workers=8,
            pin_memory=True,
        )  # note that we're passing the collate function here

    # Add Background Class
    n_classes += 1

    # Initialize model or load checkpoint
    if restarting_checkpoint_path is None:
        start_epoch = 0
        model = SSD300(n_classes=n_classes)
        # Initialize the optimizer, with twice the default learning rate for biases, as in the original Caffe repo
        biases = list()
        not_biases = list()
        for param_name, param in model.named_parameters():
            if param.requires_grad:
                if param_name.endswith(".bias"):
                    biases.append(param)
                else:
                    not_biases.append(param)
        optimizer = torch.optim.SGD(
            params=[{"params": biases, "lr": 2 * lr}, {"params": not_biases}],
            lr=lr,
            momentum=momentum,
            weight_decay=weight_decay,
        )
        # lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, 0.975)
        lr_scheduler = None
    else:
        restarting_checkpoint_path = torch.load(restarting_checkpoint_path)
        start_epoch = restarting_checkpoint_path["epoch"] + 1
        print("\nLoaded checkpoint from epoch %d.\n" % start_epoch)
        model = restarting_checkpoint_path["model"]
        optimizer = restarting_checkpoint_path["optimizer"]
        lr_scheduler = restarting_checkpoint_path.get("lr_scheduler", None)

    # Move to default device
    model = model.to(device)
    criterion = MultiBoxLoss(priors_cxcy=model.priors_cxcy).to(device)

    # Calculate total number of epochs to train and the epochs to decay learning rate at (i.e. convert iterations to epochs)
    # To convert iterations to epochs, divide iterations by the number of iterations per epoch
    # The paper trains for 120,000 iterations with a batch size of 32, decays after 80,000 and 100,000 iterations
    epochs = iterations // (len(train_loader) // 32)
    decay_lr_at = [it // (len(train_loader) // 32) for it in decay_lr_at]

    # Epochs
    for epoch in range(start_epoch, epochs):
        # Decay learning rate at particular epochs
        if epoch in decay_lr_at:
            adjust_learning_rate(optimizer, decay_lr_to)

        # One epoch's training
        losses = run_training(
            train_loader=train_loader,
            model=model,
            criterion=criterion,
            optimizer=optimizer,
            epoch=epoch,
            print_freq=print_freq,
            grad_clip=grad_clip,
        )

        if lr_scheduler is not None:
            print(f"Curr Learning Rate: {lr_scheduler.get_last_lr()}")
            lr_scheduler.step()

        # Save checkpoint
        save_checkpoint(
            epoch,
            model.state_dict(),
            optimizer,
            losses,
            lr_scheduler,
            checkpoint_output_path,
        )


def run_training(
    train_loader, model, criterion, optimizer, epoch, print_freq=200, grad_clip=None
):
    """
    One epoch's training.

    :param train_loader: DataLoader for training data
    :param model: model
    :param criterion: MultiBox loss
    :param optimizer: optimizer
    :param epoch: epoch number
    """
    model.train()  # training mode enables dropout

    batch_time = AverageMeter()  # forward prop. + back prop. time
    data_time = AverageMeter()  # data loading time
    losses = AverageMeter()  # loss
    running_loss = AverageMeter()

    start = time.time()

    # Batches
    for i, (images, boxes, labels, _) in enumerate(
        tqdm(train_loader, desc=f"Epoch {epoch}", postfix=f"Loss {losses.avg}")
    ):
        data_time.update(time.time() - start)

        # Move to default device
        images = images.to(device)  # (batch_size (N), 3, 300, 300)
        boxes = [b.to(device) for b in boxes]
        labels = [l.to(device) for l in labels]

        # Forward prop.
        predicted_locs, predicted_scores = model(
            images
        )  # (N, 8732, 4), (N, 8732, n_classes)

        # Loss
        loss = criterion(predicted_locs, predicted_scores, boxes, labels)  # scalar

        # Backward prop.
        optimizer.zero_grad()
        loss.backward()

        # Clip gradients, if necessary
        if grad_clip is not None:
            clip_gradient(optimizer, grad_clip)

        # Update model
        optimizer.step()

        losses.update(loss.item(), images.size(0))
        running_loss.update(loss.item(), images.size(0))
        batch_time.update(time.time() - start)

        start = time.time()

        # Print status
        if i % print_freq == 0:
            print(
                "Epoch: [{0}][{1}/{2}]\t"
                "Batch Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t"
                "Data Time {data_time.val:.3f} ({data_time.avg:.3f})\t"
                "Loss {loss.val:.4f} (Running {running_loss.avg:.4f})\t".format(
                    epoch,
                    i,
                    len(train_loader),
                    batch_time=batch_time,
                    data_time=data_time,
                    loss=losses,
                    running_loss=running_loss,
                )
            )
            sys.stdout.flush()
            running_loss.reset()
    del (
        predicted_locs,
        predicted_scores,
        images,
        boxes,
        labels,
    )  # free some memory since their histories may be stored
    return losses


def main():
    train_ssd(  # Learning parameters
        restarting_checkpoint_path=None,  # path to model checkpoint, None if none
        batch_size=16,  # batch size
        iterations=24000,  # number of iterations to train
        loader_workers=8,  # number of workers for loading data in the DataLoader
        print_freq=200,  # print training status every __ batches
        lr=1e-3,  # learning rate
        decay_lr_at=[16000, 30000],  # decay learning rate after these many iterations
        decay_lr_to=0.1,  # decay learning rate to this fraction of the existing learning rate
        momentum=0.9,  # momentum
        weight_decay=5e-4,  # weight decay
        grad_clip=None,  # clip if gradients are exploding, which may happen at larger batch sizes (sometimes at 32) - you will recognize it by a sorting error in the MuliBox loss calculation
    )


if __name__ == "__main__":
    main()
