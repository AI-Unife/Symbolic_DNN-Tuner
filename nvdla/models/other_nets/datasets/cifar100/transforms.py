import torchvision.transforms.v2 as transforms


def create_image_transform():
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(
                (0.5070751592371323, 0.48654887331495095, 0.4409178433670343),
                (0.2673342858792401, 0.2564384629170883, 0.27615047132568404),
            ),
        ]
    )

    return transform
