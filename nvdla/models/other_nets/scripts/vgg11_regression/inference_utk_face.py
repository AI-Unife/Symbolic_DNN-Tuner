from vision_models_pytorch.datasets.utk_face.dataset import UtkFace
from vision_models_pytorch.datasets.utk_face.transforms import data_transform
from vision_models_pytorch.models.vgg11_regression.model import vgg_11_regression
from torchinfo import summary
import torch
import torch.utils.data

from ignite.engine import create_supervised_evaluator
from ignite.metrics import MeanAbsoluteError, RootMeanSquaredError
from ignite.contrib.handlers.tqdm_logger import ProgressBar

from PIL import Image

DEFAULT_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def main(device=DEFAULT_DEVICE):
    checkpoint_path = "weights/vgg11_regression/utk_face_age/vgg11_regression_checkpoint_47_mae=5.1063.pt"
    checkpoint = torch.load(checkpoint_path)

    image_path = "test_images/test.png"

    model = vgg_11_regression()

    model.load_state_dict(checkpoint["model"])

    model.eval()
    model.to(device)

    img = Image.open(image_path).convert("RGB")
    in_tensor = data_transform()(img).unsqueeze(0).to(device)
    print(in_tensor.shape)
    with torch.no_grad():
        output = model(in_tensor)

    print(f"Age is {output.cpu().numpy().item():.1f}")


if __name__ == "__main__":
    main()
