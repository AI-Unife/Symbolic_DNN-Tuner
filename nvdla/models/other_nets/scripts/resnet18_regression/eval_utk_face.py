from vision_models_pytorch.datasets.utk_face.dataset import UtkFace
from vision_models_pytorch.datasets.utk_face.transforms import data_transform
from vision_models_pytorch.models.resnet18_regression.model import resnet18_regression
from torchinfo import summary
import torch
import torch.utils.data

from ignite.engine import create_supervised_evaluator
from ignite.metrics import MeanAbsoluteError, RootMeanSquaredError
from ignite.contrib.handlers.tqdm_logger import ProgressBar

DEFAULT_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def main(device=DEFAULT_DEVICE):
    checkpoint_path = ...
    checkpoint = torch.load(checkpoint_path)

    test_dataset = UtkFace(
        root_path="downloaded_datasets/utk_face",
        split="test",
        transform=data_transform(),
        seed=42,
    )

    test_dataloader = torch.utils.data.DataLoader(
        test_dataset, batch_size=32, pin_memory=True
    )

    model = resnet18_regression()

    model.load_state_dict(checkpoint["model"])

    model.eval()
    model.to(device)

    metrics = {
        "mae": MeanAbsoluteError(device=device),
        "rmse": RootMeanSquaredError(device=device),
    }

    pbar = ProgressBar()

    evaluator = create_supervised_evaluator(model, metrics, device=device)

    pbar.attach(evaluator)

    evaluator.run(test_dataloader)

    print(f"mae: {evaluator.state.metrics['mae']}")
    print(f"rmse: {evaluator.state.metrics['rmse']}")


if __name__ == "__main__":
    main()
