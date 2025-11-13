import torch.nn as nn
import torch.utils.data

from ignite.engine import Engine, Events
from ignite.metrics.confusion_matrix import ConfusionMatrix
from ignite.metrics import IoU, mIoU, Loss
from ignite.handlers import ModelCheckpoint
from ignite.contrib.handlers import ProgressBar

from tqdm import tqdm
from typing import Optional, Sequence

from vision_models_pytorch.models.unet.validation import make_valid_step
from vision_models_pytorch.utils.metrics import AverageMeter


DEFAULT_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def train_unet(
    model: nn.Module,
    train_dataloader: torch.utils.data.DataLoader,
    valid_dataloader: torch.utils.data.DataLoader,
    restart_checkpoint_path: Optional[str] = None,
    n_classes=3,
    class_names: Optional[Sequence[str]] = None,
    ignore_class: Optional[int] = None,
    epochs=50,
    learning_rate=1e-3,
    lr_exp_decay=0.95,
    weight_decay=1e-4,
    device=DEFAULT_DEVICE,
    checkpoint_dir="checkpoint",
    checkpoint_name="unet",
):
    """
    Trains a UNet. Validates at every epoch the UNet using mIOU. Stores in the desired path
    the two best checkpoints (evaluated at every epoch).

    Args
    ----
    - train_dataloader : torch.utils.data.DataLoader \n
        DataLoader iteratior that returns Data and Label from the Training set

    - valid_dataloader : torch.utils.data.DataLoader \n
        DataLoader iteratior that returns Data and Label from the Validation set

    - model : nn.Module \n
        The UNet model object to train from scratch.

    - restart_checkpoint_path : str | None \n
        An optional path to a checkpoint from where to resume the training process

    - n_classes : int \n
        Number of output classes of the model, that corresponds to the number of channels.

    - class_names : Optional[Sequence[str]] \n
        An optional list containing the names of the classes. It must have length equal to ``n_classes``

    - ignore_class : Optional[int] \n
        The id that corresponds to the "void" class. The class is not counted in the computation for the metrics.
        Must be between ``0`` and ``n_classes - 1``

    - epochs : int \n
        Duration of training in epochs (complete training set iterations)

    - learning_rate : float \n
        Learning rate of the Adam optimizer

    - lr_exp_decay : float \n
        Parameter for the exponential learning rate scheduler.
        Each epoch the lr is updated as lr[epoch] = lr[0] * lr_exp_decay^epoch

    - weight_decay : float \n
        L2 weight decay for regularization of the model

    - device \n
        The device where to execute the training. Defaults to GPU if available

    - checkpoint_name : str \n
        The path to the directory that stores the two best checkpoints, according to mIoU metric

    - checkpoint_prefix : str \n
        The prefix of the filename of all checkpoints


    Returns
    ---
    None
    """

    if class_names:
        assert (
            len(class_names) == n_classes
        ), "class_names must have length equal to n_classes"
    else:
        class_names = list(map(str, range(n_classes)))

    if restart_checkpoint_path is None:
        optimized_params = filter(lambda p: p.requires_grad, model.parameters())
        optimizer = torch.optim.Adam(
            optimized_params, lr=learning_rate, weight_decay=weight_decay
        )
        criterion = torch.nn.CrossEntropyLoss()
        lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(
            optimizer, lr_exp_decay, verbose=True
        )
        start_epoch = 0
    else:
        checkpoint = torch.load(restart_checkpoint_path)
        weights = checkpoint["model"]
        model.load_state_dict(weights)
        start_epoch = 0  # TODO Find a way to save the epoch number using ignite
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

    epoch = start_epoch

    model_checkpoint = ModelCheckpoint(
        checkpoint_dir,
        n_saved=2,
        filename_prefix=checkpoint_name,
        score_function=lambda engine: engine.state.metrics["mIoU"],
        score_name="accuracy",
        require_empty=False,
        global_step_transform=lambda engine, name: epoch,
    )

    handler = valid_evaluator.add_event_handler(
        Events.COMPLETED,
        model_checkpoint,
        {
            "model": model,
            "optimizer": optimizer,
            "lr_scheduler": lr_scheduler,
        },
    )

    for epoch in range(start_epoch, epochs):
        model.train()
        epoch_loss = AverageMeter()
        epoch_pbar = tqdm(
            train_dataloader, desc=f"Epoch {epoch} Training", colour="yellow"
        )
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

        valid_evaluator.run(valid_dataloader)
        metrics = valid_evaluator.state.metrics

        print(f"Validation Results - Epoch[{epoch}]\nIoU (per class):")

        for i in range(len(metrics["IoU"])):
            idx = i if ignore_class is None or i < ignore_class else i + 1
            cl_name = class_names[idx]
            iou = metrics["IoU"][i]
            print(f"{cl_name}: {iou:.3f}")

        print(f"mIoU [{metrics['mIoU']:.3f}] Val loss [{metrics['loss']:.3f}]")
