import click
import torch
from torch import nn, optim
from torchvision import transforms
from PIL import Image
import numpy as np
import os
import sys

sys.path.append('/Users/macbookpro/MLOps2024/MLOps2024/models/model1')

from model import MyAwesomeModel  # Import your model definition

def load_images_from_folder(folder_path):
    images = []
    for filename in os.listdir(folder_path):
        img_path = os.path.join(folder_path, filename)
        img = Image.open(img_path).convert('L')  # Convert to grayscale if needed
        images.append(img)
    return images

def predict(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader
) -> None:
    """Run prediction for a given model and dataloader.
    
    Args:
        model: model to use for prediction
        dataloader: dataloader with batches
    
    Returns
        Tensor of shape [N, d] where N is the number of samples and d is the output dimension of the model

    """
    # Set the model to evaluation mode
    model.eval()

    # Run prediction for a given model and dataloader
    predictions = torch.cat([model(batch) for batch in dataloader], 0)
    
    return predictions

@click.group()
def cli():
    """Command line interface."""
    pass

@click.command()
@click.argument("model_checkpoint", type=click.Path(exists=True, file_okay=True, dir_okay=False))
@click.argument("data_path", type=click.Path(exists=True, file_okay=True, dir_okay=True))
def predict_model(model_checkpoint, data_path):
    """Run prediction using a pre-trained model."""
    # Load pre-trained model
    model = MyAwesomeModel()
    checkpoint = torch.load(model_checkpoint)
    model.load_state_dict(checkpoint)
    
    # Check if CUDA is available and move the model to GPU if possible
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Load data
    if os.path.isdir(data_path):
        data = load_images_from_folder(data_path)
    elif data_path.endswith('.npy'):
        data = np.load(data_path)
    else:
        raise ValueError('Unsupported data format. Please provide a folder with raw images or a numpy file.')

    # Apply data transformation if needed
    data_transform = transforms.Compose([transforms.ToTensor(),
                                         transforms.Normalize((0.5,), (1,))])

    if isinstance(data, list):
        data = [data_transform(img) for img in data]
        dataloader = torch.utils.data.DataLoader(data, batch_size=64, shuffle=False)
    elif isinstance(data, np.ndarray):
        dataloader = torch.utils.data.DataLoader(torch.from_numpy(data), batch_size=64, shuffle=False)
    else:
        raise ValueError('Unsupported data type.')

    # Make predictions
    predictions = predict(model, dataloader)

    # Print or save predictions as needed
    print("Predictions:", predictions)

cli.add_command(predict_model)

if __name__ == "__main__":
    cli()


