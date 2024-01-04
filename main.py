import click
import torch
from torch import nn, optim
from model import MyAwesomeModel

from data import mnist


@click.group()
def cli():
    """Command line interface."""
    pass


@click.command()
@click.option("--lr", default=1e-3, help="learning rate to use for training")
def train(lr):
    """Train a model on MNIST."""
    print("Training day and night")
    print(lr)

    # TODO: Implement training loop here
    model = MyAwesomeModel()
    train_loader, _ = mnist()
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.parameters(), lr)

    epoch = 10

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
            print(f'Training loss: {running_loss/len(train_loader)}')
    
    torch.save(model.state_dict(), '/Applications/dtu_mlops/s1_development_environment/exercise_files/final_exercise/model_checkpoint.pth')

@click.command()
@click.argument("model_checkpoint")
def evaluate(model_checkpoint):
    """Evaluate a trained model."""
    print("Evaluating like my life dependends on it")
    print(model_checkpoint)

    # TODO: Implement evaluation logic here
    model = MyAwesomeModel()
    checkpoint = torch.load(model_checkpoint)
    model.load_state_dict(checkpoint)
    model.eval()
    
    _, test_loader = mnist()
    
    with torch.no_grad():
        for images, labels in test_loader:
            ps = torch.exp(model(images))
            top_p, top_class = ps.topk(1, dim=1)
            equals = top_class == labels.view(top_class.shape)
            accuracy = torch.mean(equals.type(torch.FloatTensor))
    print(f'Accuracy: {accuracy.item()*100}%')


cli.add_command(train)
cli.add_command(evaluate)


if __name__ == "__main__":
    cli()
