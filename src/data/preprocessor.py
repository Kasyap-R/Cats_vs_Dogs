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
            torch.manual_seed(self.random_seed)
            #ensure computations are deterministic, so we won't get different results each time we train
            # Can be safely removed
            torch.backends.cudnn.deterministic = True 
            torch.backends.cudnn.benchmark = False 

            np.random.shuffle(indices)
        
        # Split data into training set and validation set
        train_indices, val_indices = indices[split:], indices[:split]

        # Randomizes order of indices
        train_sampler = torch.utils.data.SubsetRandomSampler(train_indices)
        valid_sampler = torch.utils.data.SubsetRandomSampler(val_indices)

        # Creates a DataLoader, a pytorch class which serves as an iterable for efficient handling of data
        train_loader = DataLoader(dataset, batch_size=self.batch_size, sampler=train_sampler)
        validation_loader = DataLoader(dataset, batch_size=self.batch_size, sampler=valid_sampler)

        return train_loader, validation_loader

    def load_train_val(self):
        train_val_dataset = self.load_data(self.training_data_path)
        return self.split_data(train_val_dataset)

    def load_test(self):
        """Load Test Images From a Given Path and Apply Transforms"""
        dataset = UnlabeledImageFolder(self.testing_data_path, transform=self.get_transforms())
        test_loader = DataLoader(dataset, batch_size=self.batch_size)
        return test_loader
