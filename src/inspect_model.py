import torch
import os
import numpy as np
import matplotlib.pyplot as plt
from model.utils import load_config, process_image, initialize_data_loader
from model.classifier import SimpleCNN
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


def visualize_data(config_path):
    config = load_config(config_path)
    num_samples = 10
    _, axs = plt.subplots(1, num_samples, figsize=(15, 5))
    data_loader = initialize_data_loader(config)
    training_data, _ = data_loader.load_train_val()
    dataset = training_data.dataset
    
    for i in range(num_samples):
        idx = np.random.randint(0, len(dataset))
        image, label = dataset[idx]

        image = image.permute(1, 2, 0).numpy()
        image = np.clip(image, 0, 1)
        axs[i].imshow(image)
        axs[i].set_title(f"Label: {label}")
        axs[i].axis('off')
    
    plt.tight_layout()
    plt.show()


def predict(image_path, model):
    tensor = process_image(image_path)
    
    # If using GPU
    if torch.cuda.is_available():
        model.cuda()
        tensor = tensor.cuda()

    with torch.no_grad():
        outputs = model(tensor)
        # Convert logits to probabilities (for binary classification)
        probs = torch.sigmoid(outputs)
    
    probability = probs.item()

    if probability > 0.5:
        print(f"The image is classified as a Dog with probability {probability:.2f}")
    else:
        print(f"The image is classified as a Cat with probability {1 - probability:.2f}")


def gather_missed_samples(model, config, filepath: str) -> None:
    model.eval()
    train_data_path = config["data"]["train_data_path"]

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225])
    ])

    dataset = datasets.ImageFolder(train_data_path, transform=transform)
    batch_size = config["data"]["batch_size"]
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    misclassified_indices = []

    for i, (inputs, labels) in enumerate(train_loader):
        inputs = inputs.to("cuda")
        labels = labels.to("cuda")
        outputs = model(inputs)
        predicted = (torch.sigmoid(outputs) > 0.5).int()
        predicted = predicted.squeeze()
        wrong_predictions = (predicted != labels).nonzero().squeeze().tolist()
        if not isinstance(wrong_predictions, list):
            wrong_predictions = [wrong_predictions]
        misclassified_indices.extend([i * batch_size + index for index in wrong_predictions]) 

    with open(filepath, 'a') as file:
        for idx in misclassified_indices:
            path, _ = train_loader.dataset.samples[idx]
            file.write(path + '\n')


if __name__ == "__main__":
    # Deals with some random incompatibility error, IDK
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
    config_path = r"C:\Users\User\OneDrive\Documents\MachineLearning\Pet_Classifier\src\config\config.yaml"

    # Load Model Config
    config = load_config(config_path)
    model_config = config["model"]
    input_channels = model_config["input_channels"]
    hidden_units = model_config["hidden_units"]
    output_channels = model_config["output_channels"]
    batch_size = config["data"]["batch_size"]

    # Load and configure model
    model = SimpleCNN(input_channels, hidden_units, output_channels)
    model.load_state_dict(torch.load(model_config["model_save_path"]))
    model.to("cuda")
    model.eval()

    # Get a list of paths to samples the model misclassified
    # gather_missed_samples(model, config, "missed_samples.txt")

    # Check the models prediction for a certain image
    image_path = r"C:/ML_Data/Cats_vs_Dogs/train\Dogs\dog.6792.jpg"
    predict(image_path, model)




