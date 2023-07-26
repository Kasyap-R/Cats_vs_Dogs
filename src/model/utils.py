import yaml
from torchvision import transforms
from PIL import Image
from model.classifier import SimpleCNN
from data.preprocessor import ImageDatasetLoader
from torch.utils.data import DataLoader
from collections import defaultdict


def initialize_model(config):
    model = SimpleCNN(
        input_channels=config["input_channels"], 
        hidden_units=config["hidden_units"], 
        output_channels=config["output_channels"])
    return model


def initialize_data_loader(config):
    data_loader = ImageDatasetLoader(config["data"]["train_data_path"], 
                                    config["data"]["val_data_path"],
                                    config["data"]["test_data_path"], 
                                    config["data"]["batch_size"], 
                                    config["data"]["validation_split"], 
                                    config["data"]["shuffle_dataset"])
    return data_loader


def process_image(image_path: str):
    # Load Image
    img = Image.open(image_path)
    
    # Define transformations
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    # Apply transformations
    tensor = preprocess(img)
    
    # Add a batch dimension
    tensor = tensor.unsqueeze(0)
    return tensor


def load_config(config_path: str):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def get_class_distribution(dataloader: DataLoader):
    class_counts = defaultdict(int)

    for _, labels in dataloader:
        for label in labels:
            class_counts[label.item()] += 1
        
    return class_counts
