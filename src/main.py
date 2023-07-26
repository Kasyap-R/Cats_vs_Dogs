import torch
import os
from model.train import train
from model.utils import load_config, initialize_data_loader
from model.classifier import SimpleCNN
import matplotlib.pyplot as plt


if __name__ == "__main__":
    # Deals with some random incompatibility error, IDK
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
    config_path = r"C:\Users\User\OneDrive\Documents\MachineLearning\Pet_Classifier\src\config\config.yaml"
    config = load_config(config_path)
    train(config_path)

