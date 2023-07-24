import torch
import os
from model.utils import load_config, process_image, initialize_data_loader
from model.classifier import SimpleCNN
import numpy as np
import matplotlib.pyplot as plt

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
        return probs.item()


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

    # Load and configure model
    model = SimpleCNN(input_channels, hidden_units, output_channels)
    model.load_state_dict(torch.load(model_config["model_save_path"]))
    model.to("cuda")
    model.eval()

    # Make prediction
    # image_path = r"C:\ML_Data\Cats_vs_Dogs\test1\104.jpg"
    # probability = predict(image_path, model)
    # # The threshold for classification can be 0.5 (for binary classification)
    # if probability > 0.5:
    #     print(f"The image is classified as a Dog with probability {probability:.2f}")
    # else:
    #     print(f"The image is classified as a Cat with probability {1 - probability:.2f}")



