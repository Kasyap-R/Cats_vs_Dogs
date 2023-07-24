import torch.nn as nn
import torch.nn.functional as F

class SimpleCNN(nn.Module):

    def __init__(self, input_channels, hidden_units, output_channels):
        super(SimpleCNN, self).__init__()

        #Convolutional Layers
        self.conv1 = nn.Conv2d(input_channels, 16, kernel_size=3, stride=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)

        #Max Pooling Layer
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=1)

        # Assuming input image size is 224x224
        self.fc1 = nn.Linear(64 * 57 * 57, hidden_units)

        self.fc2 = nn.Linear(hidden_units, output_channels)

        #Dropout randomly stops neurons during training so they don't specialize and overfit
        self.dropout = nn.Dropout(0.5)

    # This function defines how data flows through the NN
    def forward(self, x):
        # These are the convolution layers that extract features
        # The relu function introduces non-linearization and the Max Pooling layer 
        # decreases the dimensionality by only retaining the most important features
        x = F.relu(self.conv1(x))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))

        # Flatten the tensor before feeding into the fully connected layer
        x = x.view(-1, 64 * 57 * 57)

        # Fully connected (fc) layers with ReLU activation and dropout
        # The fc layers are also called dense layers, they flatten the feature map into a vectory
        # and compute a weighted sum of the input features to arrive at an output
        x = F.relu(self.fc1(x))
        # x = self.dropout(x)
        x = self.fc2(x)

        return x
    

