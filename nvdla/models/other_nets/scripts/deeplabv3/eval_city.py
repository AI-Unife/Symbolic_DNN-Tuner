import torch
import torch.utils.data
import torch.backends.cudnn
import torchvision

from vision_models_pytorch.datasets.cityscapes.classes import (
    CITYSCAPE_CLASS_NAMES,
    CITYSCAPE_N_CLASSES,
)
from vision_models_pytorch.datasets.cityscapes.transforms import (
    create_target_transform,
    create_valid_transform,
)
from vision_models_pytorch.models.deeplabv3.validation import make_valid_step
from vision_models_pytorch.utils.augmentation import TransformableSubset

from ignite.engine import Engine
from ignite.metrics import ConfusionMatrix, mIoU
from ignite.contrib.handlers import ProgressBar


torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True

DEFAULT_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def main(
    checkpoint_path="/home/miele/vision-models-pytorch/checkpoint/cityscapes_deeplabv3_checkpoint_36_accuracy=0.7565.pt",
    device=DEFAULT_DEVICE,
):
    
    image_transform = create_valid_transform(output_size=(256, 512))

    test_dataset = torchvision.datasets.Cityscapes(
        "downloaded_datasets/cityscapes",
        mode="fine",
        target_type="semantic",
        split="val",
        transforms=image_transform,
    )
    num_classes = CITYSCAPE_N_CLASSES

    test_dataloader = torch.utils.data.DataLoader(
        test_dataset, batch_size=32, shuffle=False, num_workers=8, pin_memory=True
    )

    model = torchvision.models.segmentation.deeplabv3_resnet50(num_classes=num_classes)

    model.load_state_dict(torch.load(checkpoint_path)["model"])

    model.eval()
    model.to(device)

    valid_step = make_valid_step(model, device)
    validator = Engine(valid_step)
    conf_mat = ConfusionMatrix(num_classes, device=device)
    m_iou_metric = mIoU(conf_mat)
    m_iou_metric.attach(validator, "mIoU")

    valid_pbar = ProgressBar()
    valid_pbar.attach(validator)

    state = validator.run(test_dataloader)

    print(f'Test set mIoU = {state.metrics["mIoU"]}')


if __name__ == "__main__":
    main()
