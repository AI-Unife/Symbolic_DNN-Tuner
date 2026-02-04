
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

#########################

def getMNIST(datadir, img_size, batchsize=64, device='cpu'):

    transf = transforms.Compose([transforms.Resize(img_size), transforms.ToTensor()])

    # download and create datasets
    print(f'-I({__file__}): Loading MNIST dataset...')

    train_dataset = datasets.MNIST(root=datadir,
                                    train=True,
                                    transform=transf,
                                    download=True)

    train_dataset.train_data.to(device)
    train_dataset.train_labels.to(device)

    test_dataset = datasets.MNIST(root=datadir,
                                    train=False,
                                    transform=transf,
                                    download=True)

    test_dataset.test_data.to(device)
    test_dataset.test_labels.to(device)


    train_loader = DataLoader(dataset=train_dataset,
                            batch_size=batchsize,
                            pin_memory=True,
                            shuffle=True)

    test_loader = DataLoader(dataset=test_dataset,
                            batch_size=batchsize,
                            shuffle=False,
                            pin_memory=True
                            )


    print(f'-I({__file__}): MNIST loaded')

    return (train_loader, test_loader)
