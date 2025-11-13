from typing import Any, List, Optional, Sequence, Union
import torch
import torch.nn as nn
import torch.utils.data
import torchvision

from ignite.engine import Engine, Events
from ignite.metrics.confusion_matrix import ConfusionMatrix
from ignite.metrics import IoU, mIoU, Loss, RunningAverage
from ignite.handlers import ModelCheckpoint
from ignite.contrib.handlers import global_step_from_engine, ProgressBar

from vision_models_pytorch.models.deeplabv3.validation import make_valid_step


DEFAULT_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def train(
    train_dataloader: torch.utils.data.DataLoader,
    valid_dataloader: torch.utils.data.DataLoader,
    classes: Union[int, Sequence[str]] = 3,
    ignore_class: Optional[int] = None,
    epochs=50,
    restart_checkpoint_path=None,
    learning_rate=1e-3,
    lr_exp_decay=0.95,
    weight_decay=1e-4,
    device=DEFAULT_DEVICE,
    checkpoint_dir="checkpoint",
    checkpoint_name="deeplabv3",
):
    if isinstance(classes, list):
        n_classes = len(classes)
        class_names = classes
    else:
        n_classes = classes
        class_names = list(map(str, range(classes)))

    model = torchvision.models.segmentation.deeplabv3_resnet50(num_classes=n_classes)

    if restart_checkpoint_path is None:
        model = torchvision.models.segmentation.deeplabv3_resnet50(
            num_classes=n_classes
        )
        optimized_params = filter(lambda p: p.requires_grad, model.parameters())
        optimizer = torch.optim.Adam(
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

    valid_pbar = ProgressBar()
    valid_step = make_valid_step(model, device)
    valid_evaluator = Engine(valid_step)
    valid_pbar.attach(valid_evaluator)

    conf_matrix = ConfusionMatrix(num_classes=n_classes, device=device)
    m_iou_metric = mIoU(conf_matrix, ignore_class)
    iou_metric = IoU(conf_matrix, ignore_class)
    m_iou_metric.attach(valid_evaluator, "mIoU")
    iou_metric.attach(valid_evaluator, "IoU")

    ignite_loss = Loss(criterion, device=device)
    ignite_loss.attach(valid_evaluator, "loss")

    train_step = make_train_step(model, optimizer, optimized_params, criterion, device)
    trainer = Engine(train_step)

    trainer_pbar = ProgressBar()
    RunningAverage(output_transform=lambda x: x, device=device).attach(trainer, "loss")
    trainer_pbar.attach(trainer, ["loss"])

    epoch_complete_handler = make_epoch_complete_handler(
        valid_evaluator, valid_dataloader, lr_scheduler, class_names, ignore_class
    )
    trainer.add_event_handler(Events.EPOCH_COMPLETED, epoch_complete_handler)

    model_checkpoint = ModelCheckpoint(
        checkpoint_dir,
        n_saved=2,
        filename_prefix=checkpoint_name,
        score_function=lambda engine: engine.state.metrics["mIoU"],
        score_name="accuracy",
        require_empty=False,
        global_step_transform=global_step_from_engine(trainer),
    )

    valid_evaluator.add_event_handler(
        Events.COMPLETED,
        model_checkpoint,
        {
            "model": model,
            "optimizer": optimizer,
            "lr_scheduler": lr_scheduler,
        },
    )

    trainer.run(train_dataloader, max_epochs=epochs)


def make_train_step(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    optimized_params,
    criterion: nn.Module,
    device,
):
    def _train_step(engine, batch):
        model.train()
        model.zero_grad(set_to_none=True)

        image, label = batch
        image = image.to(device)
        label = label.to(device)

        predictions = model(image)
        loss = criterion(predictions["out"], label)

        model.zero_grad(set_to_none=True)

        loss.backward()
        optimizer.step()

        torch.nn.utils.clip_grad_norm_(optimized_params, max_norm=1.5)

        return loss.item()

    return _train_step


def make_epoch_complete_handler(
    valid_engine: Engine,
    valid_loader: torch.utils.data.DataLoader,
    lr_scheduler: torch.optim.lr_scheduler.LRScheduler,
    class_names: List[str],
    ignore_class: int,
):
    def epoch_complete_handler(engine):
        lr_scheduler.step()
        valid_engine.run(valid_loader)
        metrics = valid_engine.state.metrics

        print(f"Validation Results - Epoch[{engine.state.epoch}]\nIoU (per class):")

        for i in range(len(metrics["IoU"])):
            idx = i if ignore_class is None or i < ignore_class else i + 1
            cl_name = class_names[idx]
            iou = metrics["IoU"][i]
            print(f"{cl_name}: {iou:.3f}")

        print(f"mIoU [{metrics['mIoU']:.3f}] Val loss [{metrics['loss']:.3f}]")

    return epoch_complete_handler


if __name__ == "__main__":
    train()
