import torch
import torch.nn as nn
import torch.utils.data

from tqdm import tqdm
from vision_models_pytorch.models.vit.model import get_vit_model

from vision_models_pytorch.utils.metrics import AverageMeter, accuracy
from vision_models_pytorch.utils.training import save_checkpoint
from vision_models_pytorch.utils.augmentation import TransformableSubset

DEFAULT_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def train_vit(
    train_dataloader,
    valid_dataloader,
    n_classes=37,
    epochs=50,
    restart_checkpoint_path=None,
    learning_rate=1e-3,
    lr_exp_decay=0.95,
    weight_decay=1e-4,
    device=DEFAULT_DEVICE,
    weight_best_path="vit_train_best.pth",
    weight_latest_path="vit_train_latest.pth",
):
    if restart_checkpoint_path is None:
        model = get_vit_model(
            n_classes, pretrained_weights=True, return_transforms=False
        )
        optimized_params = filter(lambda p: p.requires_grad, model.parameters())
        optimizer = torch.optim.SGD(
            optimized_params, lr=learning_rate, weight_decay=weight_decay
        )
        criterion = torch.nn.CrossEntropyLoss()
        lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(
            optimizer, lr_exp_decay, verbose=True
        )
    else:
        checkpoint = torch.load(restart_checkpoint_path)
        model = checkpoint["model"]
        epoch = checkpoint["epoch"]
        optimizer = checkpoint["optimizer"]
        lr_scheduler = checkpoint["lr_scheduler"]

    criterion = torch.nn.CrossEntropyLoss()

    criterion.to(device)
    model.to(device)

    best_accuracy = 0.0

    for epoch in range(epochs):
        train_loss = train_one_epoch(
            model,
            train_dataloader,
            optimizer,
            lr_scheduler,
            optimized_params,
            criterion,
            epoch,
            device,
        )

        top_1, top_5, loss = validate(
            model, valid_dataloader, n_classes, epoch, criterion, device
        )

        print(f"Validation Top 1 Accuracy: {top_1:.1f}% (Best: {best_accuracy:.1f}%)")
        print(f"Validation Top 5 Accuracy: {top_5:.1f}%")
        print(f"Validation Loss: {loss:.3f}")

        if top_1 > best_accuracy:
            print(f"Overwriting best checkpoint")
            save_checkpoint(
                epoch=epoch,
                model=model,
                optimizer=optimizer,
                losses=train_loss,
                lr_scheduler=lr_scheduler,
                path=weight_best_path,
            )
            best_accuracy = top_1
        save_checkpoint(
            epoch=epoch,
            model=model,
            optimizer=optimizer,
            losses=train_loss,
            lr_scheduler=lr_scheduler,
            path=weight_latest_path,
        )


def train_one_epoch(
    model: nn.Module,
    train_dataloader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    lr_scheduler: torch.optim.lr_scheduler.LRScheduler,
    optimized_params,
    criterion: nn.Module,
    epoch: int,
    device,
):
    model.train()
    epoch_loss = AverageMeter()
    epoch_pbar = tqdm(train_dataloader, desc=f"Epoch {epoch} Training", colour="yellow")
    loss_str = "Loss {loss:.2f}"
    for image, label in epoch_pbar:
        image = image.to(device)
        label = label.to(device)

        predictions = model(image)
        loss = criterion(predictions, label)

        model.zero_grad(set_to_none=True)

        loss.backward()
        optimizer.step()

        torch.nn.utils.clip_grad_norm_(optimized_params, max_norm=1.5)

        epoch_loss.update(loss)
        epoch_pbar.postfix = loss_str.format(loss=epoch_loss.avg)

    lr_scheduler.step()

    return epoch_loss


def validate(
    model: nn.Module,
    valid_dataloader: torch.utils.data.DataLoader,
    n_classes: int,
    epoch: int,
    criterion: nn.Module,
    device,
):
    valid_pbar = tqdm(
        valid_dataloader, desc=f"Epoch {epoch} Validating", colour="green"
    )

    model.eval()

    valid_count = len(valid_dataloader)

    with torch.no_grad():
        scores = torch.zeros((valid_count, n_classes), device=device)
        labels = torch.zeros((valid_count), device=device, dtype=torch.long)

        for batch_id, (image, label) in enumerate(valid_pbar):
            image = image.to(device)
            label = label.to(device)
            predictions: torch.Tensor = model(image)
            start_idx = batch_id * valid_dataloader.batch_size
            scores[start_idx : start_idx + predictions.size(0), :] = predictions
            labels[start_idx : start_idx + label.size(0)] = label
        top_1 = accuracy(scores, labels, k=1)
        top_5 = accuracy(scores, labels, k=5)
        loss = criterion(scores, labels)

    return top_1, top_5, loss
