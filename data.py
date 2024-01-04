import torch
from torch.utils.data import DataLoader, TensorDataset

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

    # Combine images and targets into a single dataset
    train_dataset = TensorDataset(train_images, train_targets)
    test_dataset = TensorDataset(test_images, test_targets)

    # Making the train- and testloaders
    trainloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    testloader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    # Checking corruption on the data
    # Visualize a sample image from the first batch
    '''for i in range(0,15):
            sample_image = train_images[0][i, :].numpy().reshape(28, 28)  # Assuming MNIST image size
            plt.subplot(1, 15, i + 1)
            plt.imshow(sample_image, cmap='gray')
            plt.axis('off')

    plt.show()'''

    return trainloader, testloader

mnist()
'''a, b = torch.load("/Applications/dtu_mlops/data/corruptmnist/train_images_0.pt"), torch.load("/Applications/dtu_mlops/data/corruptmnist/train_target_0.pt")
print(a.size(),'\n',b.size())'''
