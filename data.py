import torch
from torch.utils.data import DataLoader, TensorDataset
from torchvision import transforms

def mnist(data_dir='/Applications/dtu_mlops/data/corruptmnist'):
    """Return train and test dataloaders for MNIST."""
    # Loading the train and test data
    train_images = [torch.load(f'{data_dir}/train_images_{i}.pt') for i in range(6)]
    train_targets = [torch.load(f'{data_dir}/train_target_{i}.pt') for i in range(6)]
    test_images = torch.load(f'{data_dir}/test_images.pt')
    test_targets = torch.load(f'{data_dir}/test_target.pt')

    # Concatenate batches along the first dimension
    train_images = torch.cat(train_images, dim=0)
    train_targets = torch.cat(train_targets, dim=0)

    # Define data augmentation transforms
    transform = transforms.Compose([
        transforms.RandomRotation(10),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    # Apply data augmentation to the training set
    trainset = TensorDataset(train_images, train_targets)
    #trainset = transforms.RandomApply([transform], p=0.5)(trainset)
    testset = TensorDataset(test_images, test_targets)

    # Combine images and targets into a single dataset
    trainloader = DataLoader(trainset, batch_size=32, shuffle=True)
    testloader = DataLoader(testset, batch_size=256, shuffle=False)

    return trainloader, testloader

a, b = torch.load("/Applications/dtu_mlops/data/corruptmnist/test_images.pt"), torch.load("/Applications/dtu_mlops/data/corruptmnist/test_target.pt")
print(a.size(),'\n',b.size()) 