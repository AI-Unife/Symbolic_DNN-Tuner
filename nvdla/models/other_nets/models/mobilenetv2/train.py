import torch
import torch.utils.data

from vision_models_pytorch.models.mobilenetv2.model import get_mobilenetv2_model

from vision_models_pytorch.utils.metrics import validate_classification
from vision_models_pytorch.utils.training import save_checkpoint, train_one_epoch

DEFAULT_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def train_mobilenetv2(
    train_dataloader,
    valid_dataloader,
    n_classes=37,
    epochs=50,
    restart_checkpoint_path=None,
    learning_rate=1e-3,
    lr_exp_decay=0.95,
    weight_decay=1e-4,
    device=DEFAULT_DEVICE,
    weight_best_path="vgg16_train_best.pth",
    weight_latest_path="vgg16_train_latest.pth",
):

    model = get_mobilenetv2_model(
        n_classes, pretrained_weights=True, return_transforms=False
    )
    optimized_params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = torch.optim.SGD(
        optimized_params, lr=learning_rate, weight_decay=weight_decay
    )
    lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(
        optimizer, lr_exp_decay, verbose=True
    )
    if restart_checkpoint_path is not None:
        checkpoint = torch.load(restart_checkpoint_path)
        model.load_state_dict(checkpoint["model"])
        epoch = checkpoint["epoch"]
        optimizer.load_state_dict(checkpoint["optimizer"])
        lr_scheduler.load_state_dict(["lr_scheduler"])

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

        top_1, top_5, loss = validate_classification(
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


