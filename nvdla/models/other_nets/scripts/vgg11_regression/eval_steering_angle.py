from vision_models_pytorch.datasets.car_steering_angle.dataset import CarSteeringAngle
from vision_models_pytorch.datasets.utk_face.transforms import (
    data_transform,
)  # Use the same transforms as UtkFace
from vision_models_pytorch.models.vgg11_regression.model import vgg_11_regression
import torch
import torch.utils.data

from ignite.engine import create_supervised_evaluator
from ignite.metrics import MeanAbsoluteError, RootMeanSquaredError
from ignite.contrib.handlers.tqdm_logger import ProgressBar

from vision_models_pytorch.utils.ignite_metrics import RegressionUsability

DEFAULT_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def main(device=DEFAULT_DEVICE):
    checkpoint_path = "checkpoints/vgg11_regression/vgg_11_regression_steering_checkpoint_50_mae=1.4743.pt"
    checkpoint = torch.load(checkpoint_path)

    test_dataset = CarSteeringAngle(
        root_path="downloaded_datasets/car_steering_angle",
        split="test",
        transform=data_transform(),
        seed=42,
    )

    test_dataloader = torch.utils.data.DataLoader(
        test_dataset, batch_size=32, pin_memory=True
    )

    model = vgg_11_regression()

    model.load_state_dict(checkpoint["model"])

    model.eval()
    model.to(device)

    metrics = {
        "mae": MeanAbsoluteError(device=device),
        "rmse": RootMeanSquaredError(device=device),
        "usab": RegressionUsability(1.0, 10.0, 0.05, device=device),
    }

    pbar = ProgressBar()

    evaluator = create_supervised_evaluator(model, metrics, device=device)

    pbar.attach(evaluator)

    evaluator.run(test_dataloader)

    print(f"mae: {evaluator.state.metrics['mae']}")
    print(f"rmse: {evaluator.state.metrics['rmse']}")
    print(f"usab: {evaluator.state.metrics['usab'] * 100}%")


if __name__ == "__main__":
    main()
