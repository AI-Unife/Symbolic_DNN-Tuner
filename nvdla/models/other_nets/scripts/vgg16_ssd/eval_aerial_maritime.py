import torch
import torch.utils.data
import torch.backends.cudnn

from vision_models_pytorch.models.vgg16_ssd.datasets.aerial_maritime import (
    AerialMaritime,
)
from vision_models_pytorch.models.vgg16_ssd.eval import evaluate_ssd
from vision_models_pytorch.models.vgg16_ssd.model import SSD300

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.benchmark = True


def main():
    dataset = AerialMaritime("downloaded_datasets/aerial_maritime", "valid")
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=16,
        collate_fn=dataset.collate_fn,
        num_workers=8,
        pin_memory=True,
    )  # note that we're passing the collate function here
    checkpoint_path = "/home/miele/vision-models-pytorch/weights/ssd/aerial_maritime/ssd_aerial_maritime_0.635_weights.pth"
    # Load model checkpoint that is to be evaluated
    weights = torch.load(checkpoint_path)

    model = SSD300(n_classes=len(dataset.get_labels()) + 1)
    model.load_state_dict(weights)
    # model = checkpoint["model"]
    model = model.to(device)

    print(f"Device: {device}")

    model.eval()

    evaluate_ssd(dataloader, model, dataset.get_labels())


if __name__ == "__main__":
    main()
