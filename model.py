import torch
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self):
<<<<<<< HEAD
        super(Net, self).__init__()

        # Define the first convolutional layer: 1 input channel, 80 output channels, 5x5 kernel
=======
        super(Network, self).__init__()
>>>>>>> 84a773764e5164adb72bfeef409122b5353c63d8
        self.conv1 = nn.Conv2d(1, 80, kernel_size=5)
        
        # Define the second convolutional layer: 80 input channels, 80 output channels, 5x5 kernel
        self.conv2 = nn.Conv2d(80, 80, kernel_size=5)
        
        # Define the first max-pooling layer: 2x2 kernel, stride 2
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        
        # Define the second max-pooling layer: 2x2 kernel, stride 2
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        
        # Batch normalization for the outputs of the first convolutional layer
        self.batch_norm1 = nn.BatchNorm2d(80)
        
        # Batch normalization for the outputs of the second convolutional layer
        self.batch_norm2 = nn.BatchNorm2d(80)
        
        # Fully connected layer: from 1280 input features to 250 output features
        self.fc1 = nn.Linear(1280, 250)
        
        # Fully connected layer: from 250 input features to 25 output features
        self.fc2 = nn.Linear(250, 25)

    def forward(self, x):
<<<<<<< HEAD
        # First convolutional layer followed by batch normalization, ReLU activation, and max-pooling
=======
>>>>>>> 84a773764e5164adb72bfeef409122b5353c63d8
        x = self.conv1(x)
        x = self.batch_norm1(x)
        x = F.relu(x)
        x = self.pool1(x)
        
        # Second convolutional layer followed by batch normalization, ReLU activation, and max-pooling
        x = self.conv2(x)
        x = self.batch_norm2(x)
        x = F.relu(x)
        x = self.pool2(x)
        
        # Flatten the tensor for the fully connected layers
        x = x.view(x.size(0), -1)
        
        # First fully connected layer followed by ReLU activation
        x = F.relu(self.fc1(x))
        
        # Second fully connected layer
        x = self.fc2(x)
        
        # Log-softmax on the output layer
        x = F.log_softmax(x, dim=1)
        
        return x

<<<<<<< HEAD
def save_model(model, model_path):
    try:
        torch.save(model.state_dict(), model_path)
        print("Model saved successfully")
    except Exception as e:
        print("Error saving model:", e)

def load_model(model_path):
    try:
        # Instantiate the model
        model = Net()
        
        # Load the state dictionary from the given path
        state_dict = torch.load(model_path, map_location=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        
        # Load the state dictionary into the model
        model.load_state_dict(state_dict)
        
        # Set the model to evaluation mode
        model.eval()
        
=======
def load_model(model_path):
    try:
        model = Network()   
        model.load_state_dict(torch.load(model_path, map_location=torch.device("cuda" if torch.cuda.is_available() else "cpu")))
        model.eval() 
>>>>>>> 84a773764e5164adb72bfeef409122b5353c63d8
        print("Model loaded successfully")
        return model
    except Exception as e:
        print("Error loading model:", e)

<<<<<<< HEAD
# Example usage:
if __name__ == "__main__":
    # Create an instance of the model
    model = Net()
    
    # Specify the path to save the model
    model_path = "model.pth"
    
    # Save the model
    save_model(model, model_path)
    
    # Load the model
    loaded_model = load_model(model_path)
=======
>>>>>>> 84a773764e5164adb72bfeef409122b5353c63d8
