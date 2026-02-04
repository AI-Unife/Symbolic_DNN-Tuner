from vision_models_pytorch.models.vgg16_ssd.datasets.kitti import Kitti
from vision_models_pytorch.models.vgg16_ssd.eval import evaluate_ssd

import torch
import torch.utils.data
import torch.backends.cudnn

from vision_models_pytorch.models.vgg16_ssd.model import SSD300

batch_size = 16
loader_workers = 8

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.benchmark = True


def main():
    test_dataset = Kitti("downloaded_datasets/kitti/training", "valid")

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=test_dataset.collate_fn,
        num_workers=loader_workers,
        pin_memory=True,
    )

    checkpoint_path = (
        "/home/miele/vision-models-pytorch/weights/ssd/kitti/ssd_kitti_1.44_weights.pth"
    )
    # Load model checkpoint that is to be evaluated
    weights = torch.load(checkpoint_path)

    model = SSD300(n_classes=len(test_dataset.get_labels()) + 1)
    model.load_state_dict(weights)

    model = model.to(device)

    print(f"Device: {device}")

    model.eval()

    evaluate_ssd(test_loader, model, test_dataset.get_labels())


if __name__ == "__main__":
    main()
