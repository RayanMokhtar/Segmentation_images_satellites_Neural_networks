from torch.utils.data import Dataset
import os
import numpy as np
import cv2
from sklearn.model_selection import train_test_split

class FloodDataset(Dataset):
    def __init__(self, image_dir, mask_dir, img_size=(256, 256), transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.img_size = img_size
        self.transform = transform
        
        self.images = os.listdir(image_dir)
        self.masks = os.listdir(mask_dir)
        
        # Ensure that the images and masks are paired correctly
        self.images.sort()
        self.masks.sort()

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.images[idx])
        mask_path = os.path.join(self.mask_dir, self.masks[idx])
        
        image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        
        # Resize images and masks
        image = cv2.resize(image, self.img_size)
        mask = cv2.resize(mask, self.img_size)
        
        if self.transform:
            image = self.transform(image)
            mask = self.transform(mask)
        
        return image, mask

def get_train_val_split(image_dir, mask_dir, test_size=0.2, random_state=42):
    images = os.listdir(image_dir)
    masks = os.listdir(mask_dir)
    
    train_images, val_images = train_test_split(images, test_size=test_size, random_state=random_state)
    train_masks, val_masks = train_test_split(masks, test_size=test_size, random_state=random_state)
    
    return train_images, val_images, train_masks, val_masks