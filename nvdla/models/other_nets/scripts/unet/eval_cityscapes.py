import torch

import torchvision

from vision_models_pytorch.datasets.cityscapes.transforms import (
    create_image_transform,
    create_valid_transform,
)

from vision_models_pytorch.datasets.cityscapes.classes import (
    CITYSCAPE_CLASS_NAMES,
    CITYSCAPE_N_CLASSES,
)

from vision_models_pytorch.models.unet.validation import make_valid_step
from vision_models_pytorch.models.unet.model import UNet

from ignite.engine import Engine
from ignite.metrics import ConfusionMatrix, mIoU, IoU
from ignite.contrib.handlers import ProgressBar


torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True

DEFAULT_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def main(
    checkpoint_path="weights/unet/cityscapes/unet_cityscapes_checkpoint_43_accuracy=0.7095.pt",
    device=DEFAULT_DEVICE,
):
    valid_transform = create_valid_transform(output_size=(128, 256))
    # Dataset download (if not already downloaded)
    test_dataset = torchvision.datasets.Cityscapes(
        "downloaded_datasets/cityscapes",
        mode="fine",
        target_type="semantic",
        split="val",
        transforms=valid_transform,
    )

    num_classes = CITYSCAPE_N_CLASSES
    ignore_class = 0
    test_dataloader = torch.utils.data.DataLoader(
        test_dataset, batch_size=32, shuffle=False, num_workers=8, pin_memory=True
    )

    model = UNet(
        in_channels=3, n_classes=CITYSCAPE_N_CLASSES, padding=True, batch_norm=True
    )
    # Load weights
    model.load_state_dict(torch.load(checkpoint_path)["model"])

    model.eval()  # Set evaluation mode
    model.to(device)  # Move model to device (GPU)

    valid_step = make_valid_step(model, device)  # Get validation iteration function
    validator = Engine(valid_step)  # Create ignite Engine for validation
    conf_mat = ConfusionMatrix(
        num_classes, device=device
    )  # Create confusion matrix needed for mIoU computation
    m_iou_metric = mIoU(conf_mat, ignore_class)
    iou_metric = IoU(conf_mat, ignore_class)
    m_iou_metric.attach(validator, "mIoU")
    iou_metric.attach(validator, "IoU")
    # Register Progress Bar rendering
    valid_pbar = ProgressBar()
    valid_pbar.attach(validator)

    # Start validation
    state = validator.run(test_dataloader)

    print(f'Test set mIoU = {state.metrics["IoU"]}')
    print(f'Test set mIoU = {state.metrics["mIoU"]}')


if __name__ == "__main__":
    main()
