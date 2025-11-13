from typing import Optional
from ignite.engine import Events, create_supervised_trainer, create_supervised_evaluator
from ignite.metrics import MeanAbsoluteError, RunningAverage
from ignite.handlers.checkpoint import ModelCheckpoint
from ignite.contrib.handlers.tqdm_logger import ProgressBar
from ignite.contrib.handlers import global_step_from_engine
import torch
import torch.nn as nn
import torch.utils.data

DEFAULT_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def train_resnet18_regression(
    model: nn.Module,
    train_dataloader: torch.utils.data.DataLoader,
    valid_dataloader: torch.utils.data.DataLoader,
    resume_checkpoint_path: Optional[str] = None,
    learning_rate=3e-4,
    learning_rate_decay=0.95,
    weight_decay=0.005,
    epochs=50,
    device=DEFAULT_DEVICE,
    checkpoint_dir="checkpoints/resnet18_regression",
    checkpoint_prefix="resnet18_regression",
):
    """
    Trains ResNet18 Regression model
    """
    criterion = torch.nn.L1Loss()
    optimized_params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = torch.optim.Adam(
        optimized_params, lr=learning_rate, weight_decay=weight_decay
    )
    lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(
        optimizer, learning_rate_decay
    )

    if resume_checkpoint_path:
        checkpoint = torch.load(resume_checkpoint_path)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])

    model.to(device)
    criterion.to(device)

    val_metrics = {"mae": MeanAbsoluteError()}

    trainer = create_supervised_trainer(model, optimizer, criterion, device)
    evaluator = create_supervised_evaluator(model, val_metrics, device)

    trainer_pbar = ProgressBar()

    model_checkpoint = ModelCheckpoint(
        checkpoint_dir,
        n_saved=None,
        filename_prefix=checkpoint_prefix,
        score_function=lambda engine: engine.state.metrics["mae"],
        score_name="mae",
        require_empty=False,
        global_step_transform=global_step_from_engine(trainer),
    )
    evaluator.add_event_handler(
        Events.COMPLETED,
        model_checkpoint,
        {"model": model, "optimizer": optimizer, "lr_scheduler": lr_scheduler},
    )

    avg_loss = RunningAverage(output_transform=lambda x: x)
    avg_loss.attach(trainer, "avg_loss")

    trainer_pbar.attach(trainer, ["avg_loss"])
    trainer_pbar.attach(evaluator)

    @trainer.on(Events.EPOCH_COMPLETED)
    def log_training_loss(engine):
        lr_scheduler.step()
        evaluator.run(valid_dataloader)
        metrics = evaluator.state.metrics
        print(
            f"Training Results - Epoch[{trainer.state.epoch}] Avg MAE: {metrics['mae']:.2f}"
        )

    trainer.run(train_dataloader, max_epochs=epochs)


if __name__ == "__main__":
    train_resnet18_regression()
