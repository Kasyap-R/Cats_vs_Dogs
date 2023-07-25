import torch
import torch.nn as nn
import torch.optim as optim
from model.utils import load_config, initialize_data_loader, initialize_model
from tqdm import tqdm
from torch.optim.lr_scheduler import StepLR


def train_batch(model, batch, loss_function, optimizer, device):
    model.train()  # Ensure the model is in training mode
    
    images, labels = batch  # Unpack the batch
    images, labels = images.to(device), labels.to(device)
    outputs = model(images).squeeze(1)  # Forward pass and squeeze to adjust dimensions
    labels = labels.float()
    
    loss = loss_function(outputs, labels)
    
    optimizer.zero_grad()  # Zero the optimizer gradients
    loss.backward()  # Backward pass
    optimizer.step()  # Update the weights
    
    # For binary classification
    predicted = (torch.sigmoid(outputs) > 0.5).float()  # Convert to probabilities and then to 0 or 1.
    correct = (predicted == labels).sum().item()
    
    return loss.item(), correct


def validate_epoch(model, validation_data, loss_function, device):
    model.eval() # Sets model to eval mode
    total_val_loss = 0.0
    correct_predictions = 0
    total_samples = 0

    with torch.no_grad(): # Keeps tells it to not track gradients as we're only validating
        for images, labels in validation_data:
            images, labels = images.to(device), labels.to(device).float().unsqueeze(1)
            outputs = model(images)
            loss = loss_function(outputs, labels)
            total_val_loss += loss.item()
            predicted = (torch.sigmoid(outputs) > 0.5).float()
            correct_predictions += (predicted == labels).sum().item()
            total_samples += labels.size(0)

    average_val_loss = total_val_loss / len(validation_data)
    accuracy = 100 * correct_predictions / total_samples

    return average_val_loss, accuracy


def train(config_path):
    config = load_config(config_path)

    # Load Data
    data_loader = initialize_data_loader(config)
    training_data, validation_data = data_loader.load_train_val()

    # Initialize Model and move to appropriate device (either CPU or GPY)
    model_config = config["model"]
    model = initialize_model(model_config)
    device = torch.device("cuda") 
    model.to(device)

    # Define loss functions and optimizer (which just minimzes the loss function)
    loss_function = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=config['training']['learning_rate'])
    scheduler = StepLR(optimizer, step_size=5, gamma=0.1)  # Reduces the learning rate by a factor of 0.1 every 5 epochs.

    # Training Loop
    num_epochs = config["training"]["epochs"]
    for epoch in range(num_epochs):
        total_train_loss = 0.0
        total_train_correct = 0
        
        # Training phase
        for batch in tqdm(training_data, desc=f"Epoch {epoch+1}"):
            batch_loss, batch_correct = train_batch(model, batch, loss_function, optimizer, device)
            total_train_loss += batch_loss
            total_train_correct += batch_correct
        
        average_train_loss = total_train_loss / len(training_data)
        train_accuracy = 100.0 * total_train_correct / len(training_data.dataset)
        
        # Validation phase
        average_val_loss, val_accuracy = validate_epoch(model, validation_data, loss_function, device)
        
        print(f"Epoch [{epoch+1}/{num_epochs}]")
        print(f"Training Loss: {average_train_loss:.4f}\t Training Accuracy: {train_accuracy:.2f}%")
        print(f"Validation Loss: {average_val_loss:.4f}\t Validation Accuracy: {val_accuracy:.2f}%\n")

        # Decreases learning rate when it has 'stepped' 5 times
        scheduler.step()

    # Save the model
    torch.save(model.state_dict(), model_config["model_save_path"])

