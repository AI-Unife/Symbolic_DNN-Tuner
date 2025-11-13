from vision_models_pytorch.models.vgg16_ssd.datasets.kitti import Kitti
from vision_models_pytorch.models.vgg16_ssd.train import train_ssd

import torch


def main():
    train_dataset = Kitti(split="train")

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=8,
        shuffle=True,
        collate_fn=train_dataset.collate_fn,
        num_workers=8,
        pin_memory=True,
    )  # note that we're passing the collate function here
    train_ssd(
        train_loader=train_loader,
        restarting_checkpoint_path="chkpt_kitti_ssd300_105_1.93.pth.tar",
        checkpoint_output_path="chkpt_kitti_ssd300_{epoch}_{loss:.2f}.pth.tar",
        print_freq=100,
        grad_clip=True,
        iterations=108000,
        n_classes=len(train_dataset.get_labels()),
        decay_lr_at=[6000, 10000, 14000, 18000, 22000, 26000, 30000],
        decay_lr_to=0.25,
        lr=1e-5,
    )


if __name__ == "__main__":
    main()
