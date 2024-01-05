import sys
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt

sys.path.append('/Users/macbookpro/MLOps2024/MLOps2024/models/model1')

from model import MyAwesomeModel

def train(lr):
    """Train a model on MNIST."""
    
    lr = float(lr)
    
    print(lr)

    # TODO: Implement training loop here
    model = MyAwesomeModel()
    images = torch.load('/Users/macbookpro/MLOps2024/MLOps2024/data/processed/processed_images.pt')
    targets = torch.load('/Users/macbookpro/MLOps2024/MLOps2024/data/processed/processed_targets.pt')
    trainset = TensorDataset(images, targets)
    train_loader = DataLoader(trainset, batch_size=64, shuffle=True)
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.parameters(), lr)

    epoch = 10

    losses = []

    for e in range(epoch):
        
        model.train()
        
        running_loss = 0

        for images, labels in train_loader:
            images = images.view(images.shape[0], -1)

            optimizer.zero_grad()
            output = model(images)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
        else:
            average_loss = running_loss/len(train_loader)
            losses.append(average_loss)
            print(f'Training loss: {average_loss}')
    
    torch.save(model.state_dict(), '/Users/macbookpro/MLOps2024/MLOps2024/models/model1/model_checkpoint.pth')

    plt.plot(range(1, epoch+1), losses, marker='o')
    plt.xlabel('Epoch')
    plt.ylabel('Training Loss')
    plt.title(f'Training Loss Over {e+1} Epochs')
    plt.savefig('/Users/macbookpro/MLOps2024/MLOps2024/reports/figures')

train(input('Learning rate:\n'))

if __name__ == "__main__":
    pass
