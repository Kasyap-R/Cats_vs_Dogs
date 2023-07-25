import torch 
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split, Dataset
import numpy as np
from PIL import Image
import os

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


class SimpleImageDataset(Dataset):
    def __init__(self, samples):
        """
        samples: List of (image_tensor, label) tuples
        """
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]



class ImageDatasetLoader():
    def __init__(self, training_data_path: str, testing_data_path: str, batch_size: int, validation_split: float, shuffle_dataset: bool):
        self.training_data_path = training_data_path
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


    def split_data(self, dataset):
        """Splits Training Data into Training and Validation Set"""
        dataset_size = len(dataset)
        indices = list(range(dataset_size))
        split = int(self.validation_split * dataset_size)
        
        if self.shuffle_dataset:
            np.random.shuffle(indices)
        
        # Split data into training set and validation set
        train_indices, val_indices = indices[split:], indices[:split]
        
        train_samples = [dataset[i] for i in train_indices]
        val_samples = [dataset[i] for i in val_indices]

        # Create SimpleImageDataset instances
        train_dataset = SimpleImageDataset(train_samples)
        val_dataset = SimpleImageDataset(val_samples)

        print("Size of the dataset before oversampling:", len(train_samples))
        train_dataset = self.oversample_hard_data(train_dataset)  # Modify the oversample_hard_data method accordingly

        # Randomizes order of indices
        train_sampler = torch.utils.data.SubsetRandomSampler(train_indices)
        valid_sampler = torch.utils.data.SubsetRandomSampler(val_indices)

        # Create DataLoader instances
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, sampler=train_sampler)
        validation_loader = DataLoader(val_dataset, batch_size=self.batch_size, sampler=valid_sampler)

        return train_loader, validation_loader
    

    def oversample_hard_data(self, train_samples):
        # Define the augmentation pipeline
        augmentation = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            # ... add other augmentation techniques as needed
        ])

        # Read misclassified samples
        with open("missed_samples.txt", 'r') as f:
            missed_samples = [line.strip() for line in f.readlines()]

        augmented_samples = []
        for path, label in train_samples:
            if path in missed_samples:
                image = Image.open(path)
                augmented_image = augmentation(image)
                augmented_image = self.get_transforms()(augmented_image)  
                augmented_tensor = transforms.ToTensor()(augmented_image)
                augmented_samples.append((path, label))  

        # Extend the training samples
        train_samples.extend(augmented_samples)
        return train_samples

    def load_train_val(self):
        train_val_dataset = self.load_data(self.training_data_path)
        return self.split_data(train_val_dataset)

    def load_test(self):
        """Load Test Images From a Given Path and Apply Transforms"""
        dataset = UnlabeledImageFolder(self.testing_data_path, transform=self.get_transforms())
        test_loader = DataLoader(dataset, batch_size=self.batch_size)
        return test_loader
