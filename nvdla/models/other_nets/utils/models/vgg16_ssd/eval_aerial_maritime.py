import torch

from vision_models_pytorch.models.vgg16_ssd.datasets.aerial_maritime import (
    AerialMaritime,
)
from vision_models_pytorch.models.vgg16_ssd.eval import evaluate_ssd

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
    checkpoint_path = "weights/checkpoint_boats_ssd300_0635.pth.tar"
    # Load model checkpoint that is to be evaluated
    checkpoint = torch.load(checkpoint_path)
    model = checkpoint["model"]
    model = model.to(device)

    print(f"Device: {device}")

    model.eval()

    evaluate_ssd(dataloader, model, dataset.get_labels())


if __name__ == "__main__":
    main()
