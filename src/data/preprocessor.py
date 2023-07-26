import torch 
import os
import numpy as np
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split, Dataset
from PIL import Image
from torch.utils.data import Subset

# Custom Implementation of PyTorch's Datset class to support 
# the images in our test class, which are unlabeled
class UnlabeledImageFolder(Dataset):
    def __init__(self, path, transform=None):
        self.path = path
        self.image_files = os.listdir(path)
        self.transform = transform

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, index):
        """Determines how indexing will work with our test dataset"""
        image = Image.open(os.path.join(self.path, self.image_files[index]))
        if self.transform:
            image = self.transform(image)
        return image


# Our wrapper around the dataset class because we can't directly add images to the dataset returned by datasets.ImageFolder()
# so we turn it into a list and add it to the augmented images and THEN turn that combined list BACK into a dataset
class CustomImageDataset(Dataset):
    def __init__(self, data):
        # data should be a list of (image, label) pairs
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


class ImageDatasetLoader():


    def __init__(self, training_data_path: str, val_data_path: str, testing_data_path: str, batch_size: int, validation_split: float, shuffle_dataset: bool):
        self.training_data_path = training_data_path
        self.val_data_path = val_data_path
        self.testing_data_path = testing_data_path
        self.batch_size = batch_size
        self.validation_split = validation_split
        self.shuffle_dataset = shuffle_dataset
        self.random_seed = 42


    def get_transforms(self):
        """Define transformations for the images"""
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
        return transform


    def load_data(self, path: str):
        """Load Images From a Given Path and Apply Transforms"""
        dataset = datasets.ImageFolder(path, transform=self.get_transforms())
        return dataset


    def load_train(self):
        train_dataset = self.load_data(self.training_data_path)
        print(f"Length of dataset before augmentation {len(train_dataset)}")
        train_dataset = self.oversample_hard_images(train_dataset)
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=self.shuffle_dataset)
        return train_loader
    

    def load_val(self):
        val_dataset = self.load_data(self.val_data_path)
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=self.shuffle_dataset)
        return val_loader


    def load_test(self):
        """Load Test Images From a Given Path and Apply Transforms"""
        dataset = UnlabeledImageFolder(self.testing_data_path, transform=self.get_transforms())
        test_loader = DataLoader(dataset, batch_size=self.batch_size)
        return test_loader
    

    def robust_augmentation_pipeline(self):
        """Defines a robust augmentation pipeline for images."""
        transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(30),
            transforms.RandomResizedCrop(224, scale=(0.8, 1.0), antialias=True),  # Assuming image size is 224x224, adjust if needed
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        ])
        return transform


    def oversample_hard_images(self, dataset: Dataset) -> CustomImageDataset:
        """
        Given a dataset, augment all the images that it has given
        """
        with open("missed_samples.txt", 'r') as f:
            missed_paths = f.read().splitlines()
        
        augmented_samples = []
        
        for path in missed_paths:
            for sample in dataset.samples:
                if path == sample[0]:
                    image, label = sample
                    pil_image = Image.open(image)
                    transformed_image = self.get_transforms()(pil_image)
                    aug_transform = self.robust_augmentation_pipeline()
                    augmented_image = aug_transform(transformed_image)
                    augmented_samples.append((augmented_image, label))
                    break
        
        extended_data = list(dataset) + augmented_samples
        print(f"Length of dataset after augmentation {len(extended_data)}")
        return CustomImageDataset(extended_data)

