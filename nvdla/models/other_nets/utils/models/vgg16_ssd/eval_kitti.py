from vision_models_pytorch.models.vgg16_ssd.datasets.kitti import Kitti
from vision_models_pytorch.models.vgg16_ssd.eval import evaluate_ssd

import torch
import torch.utils.data
import torch.backends.cudnn

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

    checkpoint_path = "weights/ssd/kitti/chkpt_kitti_ssd300_230_1.44.pth.tar"
    # Load model checkpoint that is to be evaluated
    checkpoint = torch.load(checkpoint_path)
    model = checkpoint["model"]
    model = model.to(device)

    print(f"Device: {device}")

    model.eval()

    evaluate_ssd(test_loader, model, test_dataset.get_labels())


if __name__ == "__main__":
    main()
