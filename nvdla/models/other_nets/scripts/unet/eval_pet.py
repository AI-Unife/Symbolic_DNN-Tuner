import torch

import torchvision

from vision_models_pytorch.datasets.oxford_iiit_pet.transforms import (
    create_image_transform,
    create_target_transform,
)
from vision_models_pytorch.models.unet.validation import make_valid_step
from vision_models_pytorch.models.unet.model import UNet
from vision_models_pytorch.utils.augmentation import TransformableSubset

from ignite.engine import Engine
from ignite.metrics import ConfusionMatrix, mIoU
from ignite.contrib.handlers import ProgressBar

from operator import itemgetter

torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True

DEFAULT_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def main(
    checkpoint_path="weights/unet/pet/unet_pet_0.6707.pt",
    device=DEFAULT_DEVICE,
):
    # Dataset download (if not already downloaded)
    data = torchvision.datasets.OxfordIIITPet(
        "downloaded_datasets/oxford_iiit_pet",
        download=True,
        split="test",
        target_types="segmentation",
        target_transform=create_target_transform(),
    )

    num_classes = 3
    # Apply transformation to all images
    test_dataset = TransformableSubset(data, data_transform=create_image_transform())
    test_dataloader = torch.utils.data.DataLoader(
        test_dataset, batch_size=32, shuffle=False, num_workers=8, pin_memory=True
    )

    model = UNet(in_channels=3, n_classes=3, padding=True)

    checkpoint = torch.load(checkpoint_path)['model']

    # Load weights
    model.load_state_dict(checkpoint)

    model.eval()  # Set evaluation mode
    model.to(device)  # Move model to device (GPU)

    valid_step = make_valid_step(model, device)  # Get validation iteration function
    validator = Engine(valid_step)  # Create ignite Engine for validation
    conf_mat = ConfusionMatrix(
        num_classes, device=device
    )  # Create confusion matrix needed for mIoU computation
    m_iou_metric = mIoU(conf_mat)
    m_iou_metric.attach(validator, "mIoU")  # Register mIoU computation to validator

    # Register Progress Bar rendering
    valid_pbar = ProgressBar()
    valid_pbar.attach(validator)

    # Start validation
    state = validator.run(test_dataloader)

    print(f'Test set mIoU = {state.metrics["mIoU"]}')


if __name__ == "__main__":
    main()
