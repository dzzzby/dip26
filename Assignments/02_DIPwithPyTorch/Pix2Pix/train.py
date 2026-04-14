import os
import argparse
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from facades_dataset import FacadesDataset
from FCN_network import FullyConvNetwork
from torch.optim.lr_scheduler import StepLR

def tensor_to_image(tensor):
    """
    Convert a PyTorch tensor to a NumPy array suitable for OpenCV.

    Args:
        tensor (torch.Tensor): A tensor of shape (C, H, W).

    Returns:
        numpy.ndarray: An image array of shape (H, W, C) with values in [0, 255] and dtype uint8.
    """
    # Move tensor to CPU, detach from graph, and convert to NumPy array
    image = tensor.cpu().detach().numpy()
    # Transpose from (C, H, W) to (H, W, C)
    image = np.transpose(image, (1, 2, 0))
    # Denormalize from [-1, 1] to [0, 1]
    image = (image + 1) / 2
    # Scale to [0, 255] and convert to uint8
    image = (image * 255).astype(np.uint8)
    return image

def save_images(inputs, targets, outputs, folder_name, epoch, num_images=5):
    """
    Save a set of input, target, and output images for visualization.

    Args:
        inputs (torch.Tensor): Batch of input images.
        targets (torch.Tensor): Batch of target images.
        outputs (torch.Tensor): Batch of output images from the model.
        folder_name (str): Directory to save the images ('train_results' or 'val_results').
        epoch (int): Current epoch number.
        num_images (int): Number of images to save from the batch.
    """
    os.makedirs(f'{folder_name}/epoch_{epoch}', exist_ok=True)
    n = min(num_images, inputs.shape[0])
    for i in range(n):
        # Convert tensors to images
        input_img_np = tensor_to_image(inputs[i])
        target_img_np = tensor_to_image(targets[i])
        output_img_np = tensor_to_image(outputs[i])

        # Concatenate the images horizontally
        comparison = np.hstack((input_img_np, target_img_np, output_img_np))

        # Save the comparison image
        cv2.imwrite(f'{folder_name}/epoch_{epoch}/result_{i + 1}.png', comparison)

def train_one_epoch(model, dataloader, optimizer, criterion, device, epoch, num_epochs, scaler, log_interval, save_every):
    """
    Train the model for one epoch.

    Args:
        model (nn.Module): The neural network model.
        dataloader (DataLoader): DataLoader for the training data.
        optimizer (Optimizer): Optimizer for updating model parameters.
        criterion (Loss): Loss function.
        device (torch.device): Device to run the training on.
        epoch (int): Current epoch number.
        num_epochs (int): Total number of epochs.
    """
    model.train()
    running_loss = 0.0

    use_amp = scaler is not None
    for i, (image_input, image_target) in enumerate(dataloader):
        # Move data to the device
        image_input = image_input.to(device, non_blocking=True)
        image_target = image_target.to(device, non_blocking=True)

        # Zero the gradients
        optimizer.zero_grad()

        # Forward pass
        with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=use_amp):
            outputs = model(image_input)
            # Compute the loss
            loss = criterion(outputs, image_target)

        # Save sample images every 5 epochs
        if (epoch + 1) % save_every == 0 and i == 0:
            save_images(image_input, image_target, outputs, 'train_results', epoch + 1)

        # Backward pass and optimization
        if use_amp:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        # Update running loss
        running_loss += loss.item()

        # Print loss information
        if (i + 1) % log_interval == 0 or i == 0 or (i + 1) == len(dataloader):
            print(f'Epoch [{epoch + 1}/{num_epochs}], Step [{i + 1}/{len(dataloader)}], Loss: {loss.item():.4f}')

def validate(model, dataloader, criterion, device, epoch, num_epochs, save_every):
    """
    Validate the model on the validation dataset.

    Args:
        model (nn.Module): The neural network model.
        dataloader (DataLoader): DataLoader for the validation data.
        criterion (Loss): Loss function.
        device (torch.device): Device to run the validation on.
        epoch (int): Current epoch number.
        num_epochs (int): Total number of epochs.
    """
    model.eval()
    val_loss = 0.0

    with torch.no_grad():
        for i, (image_input, image_target) in enumerate(dataloader):
            # Move data to the device
            image_input = image_input.to(device, non_blocking=True)
            image_target = image_target.to(device, non_blocking=True)

            # Forward pass
            outputs = model(image_input)

            # Compute the loss
            loss = criterion(outputs, image_target)
            val_loss += loss.item()

            # Save sample images every 5 epochs
            if (epoch + 1) % save_every == 0 and i == 0:
                save_images(image_input, image_target, outputs, 'val_results', epoch + 1)

    # Calculate average validation loss
    avg_val_loss = val_loss / len(dataloader)
    print(f'Epoch [{epoch + 1}/{num_epochs}], Validation Loss: {avg_val_loss:.4f}')

def parse_args():
    parser = argparse.ArgumentParser(description='Train Pix2Pix FCN on facades dataset.')
    parser.add_argument('--direction', choices=['left2right', 'right2left'], default='right2left')
    parser.add_argument('--image-size', type=int, default=256)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--num-workers', type=int, default=0 if os.name == 'nt' else 4)
    parser.add_argument('--lr', type=float, default=2e-4)
    parser.add_argument('--step-size', type=int, default=60)
    parser.add_argument('--gamma', type=float, default=0.5)
    parser.add_argument('--save-every', type=int, default=20)
    parser.add_argument('--log-interval', type=int, default=20)
    parser.add_argument('--weight-decay', type=float, default=1e-4)
    return parser.parse_args()


def main():
    """
    Main function to set up the training and validation processes.
    """
    args = parse_args()

    # Set device to GPU if available
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    if device.type == 'cuda':
        torch.backends.cudnn.benchmark = True

    # Initialize datasets and dataloaders
    train_dataset = FacadesDataset(
        list_file='train_list.txt',
        direction=args.direction,
        augment=True,
        image_size=args.image_size,
    )
    val_dataset = FacadesDataset(
        list_file='val_list.txt',
        direction=args.direction,
        augment=False,
        image_size=args.image_size,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device.type == 'cuda'),
        persistent_workers=(args.num_workers > 0),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == 'cuda'),
        persistent_workers=(args.num_workers > 0),
    )

    # Initialize model, loss function, and optimizer
    model = FullyConvNetwork().to(device)
    criterion = nn.L1Loss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=(0.5, 0.999), weight_decay=args.weight_decay)
    scaler = torch.amp.GradScaler('cuda', enabled=(device.type == 'cuda'))

    # Add a learning rate scheduler for decay
    scheduler = StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)

    # Training loop
    num_epochs = args.epochs
    for epoch in range(num_epochs):
        train_one_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
            device,
            epoch,
            num_epochs,
            scaler,
            args.log_interval,
            args.save_every,
        )
        validate(model, val_loader, criterion, device, epoch, num_epochs, args.save_every)

        # Step the scheduler after each epoch
        scheduler.step()

        # Save model checkpoint every 50 epochs
        if (epoch + 1) % args.save_every == 0:
            os.makedirs('checkpoints', exist_ok=True)
            torch.save(model.state_dict(), f'checkpoints/pix2pix_model_epoch_{epoch + 1}.pth')

if __name__ == '__main__':
    main()
