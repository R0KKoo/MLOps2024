if __name__ == '__main__':
    # Get the data and process it
    import torch
    from torchvision import transforms
    import os
    
    raw_path = '/Applications/dtu_mlops/data/corruptmnist'
    processed_path = '/Users/macbookpro/MLOps2024/MLOps2024/data/processed'

    # Load and concatenate data
    train_images = [torch.load(f'{raw_path}/train_images_{i}.pt') for i in range(6)]
    train_targets = [torch.load(f'{raw_path}/train_target_{i}.pt') for i in range(6)]

    images_processed = torch.cat(train_images, dim=0)
    targets_processed = torch.cat(train_targets, dim=0)

    # Convert PyTorch tensor to NumPy array
    images_numpy = images_processed.numpy()

    # Add channel dimension to the NumPy array
    images_numpy = images_numpy.reshape(-1, 28, 28, 1)

    # Convert NumPy array to PyTorch tensor and apply data transformation with normalization
    im_p = transforms.functional.normalize(torch.from_numpy(images_numpy), mean=(0.5,), std=(1,))

    # Save processed data
    torch.save(im_p, os.path.join(processed_path, 'processed_images.pt'))
    torch.save(targets_processed, os.path.join(processed_path, 'processed_targets.pt'))

    print(im_p[0], im_p.size())

    pass
