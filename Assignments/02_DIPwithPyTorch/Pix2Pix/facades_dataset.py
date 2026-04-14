import torch
from torch.utils.data import Dataset
import cv2
import random

class FacadesDataset(Dataset):
    def __init__(self, list_file, direction='right2left', augment=False, image_size=256):
        """
        Args:
            list_file (string): Path to the txt file with image filenames.
            direction (string): 'left2right' or 'right2left'.
            augment (bool): Whether to apply paired augmentation on training data.
            image_size (int): Resize each side of paired image to image_size x image_size.
        """
        # Read the list of image filenames
        with open(list_file, 'r') as file:
            self.image_filenames = [line.strip() for line in file]
        if direction not in ('left2right', 'right2left'):
            raise ValueError("direction must be 'left2right' or 'right2left'")
        self.direction = direction
        self.augment = augment
        self.image_size = image_size
        
    def __len__(self):
        # Return the total number of images
        return len(self.image_filenames)
    
    def __getitem__(self, idx):
        # Get the image filename
        img_name = self.image_filenames[idx]
        img_color_semantic = cv2.imread(img_name)
        img_color_semantic = cv2.cvtColor(img_color_semantic, cv2.COLOR_BGR2RGB)
        half_w = img_color_semantic.shape[1] // 2
        image_left = img_color_semantic[:, :half_w, :]
        image_right = img_color_semantic[:, half_w:half_w * 2, :]

        if self.augment and random.random() < 0.5:
            image_left = cv2.flip(image_left, 1)
            image_right = cv2.flip(image_right, 1)

        if self.image_size is not None and self.image_size > 0:
            target_size = (self.image_size, self.image_size)
            image_left = cv2.resize(image_left, target_size, interpolation=cv2.INTER_LINEAR)
            image_right = cv2.resize(image_right, target_size, interpolation=cv2.INTER_LINEAR)

        if self.direction == 'left2right':
            image_input = image_left
            image_target = image_right
        else:
            image_input = image_right
            image_target = image_left

        # Convert to tensor in [-1, 1]
        image_input = torch.from_numpy(image_input).permute(2, 0, 1).float() / 255.0 * 2.0 - 1.0
        image_target = torch.from_numpy(image_target).permute(2, 0, 1).float() / 255.0 * 2.0 - 1.0

        return image_input, image_target