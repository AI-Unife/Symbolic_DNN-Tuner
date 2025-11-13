import os

from torchvision import datasets, transforms
from torch.utils.data import DataLoader

#########################
datasetdir = os.environ['TORCH_DATASETDIR']


def getCIFAR(datasetdir, img_size=32, batchsize=64, device='cpu'):

    transf = transforms.Compose([transforms.Resize(img_size), transforms.ToTensor()])

    # download and create datasets
    print(f'-I({__file__}): Loading Cifar 10 dataset...')

    train_dataset = datasets.CIFAR10(root=datasetdir,
                                     train=True,
                                     transform=transf,
                                     download=True)


    test_dataset = datasets.CIFAR10(root=datasetdir,
                                    train=False,
                                    transform=transf,
                                    download=True)



    train_loader = DataLoader(dataset=train_dataset,
                              batch_size=batchsize,
                              pin_memory=True,
                              shuffle=True,
                              drop_last=True)

    print(batchsize)

    test_loader = DataLoader(dataset=test_dataset,
                             batch_size=batchsize,
                             shuffle=False,
                             pin_memory=True,
                             drop_last=True
                             )


    print(f'-I({__file__}): CIFAR 10 loaded')

    return (train_loader, test_loader)