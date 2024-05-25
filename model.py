import torch
import torch.nn as nn
import torch.nn.functional as F

class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()
        
        # Define the first convolutional layer: 1 input channel, 16 output channels, 3x3 kernel
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        
        # Define the second convolutional layer: 16 input channels, 32 output channels, 3x3 kernel
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        
        # Define the first max-pooling layer: 2x2 kernel, stride 2
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        
        # Define the second max-pooling layer: 2x2 kernel, stride 2
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        
        # Batch normalization for the outputs of the first convolutional layer
        self.batch_norm1 = nn.BatchNorm2d(16)
        
        # Batch normalization for the outputs of the second convolutional layer
        self.batch_norm2 = nn.BatchNorm2d(32)
        
        # Dropout layer with 0.5 probability
        self.dropout = nn.Dropout(p=0.5)
        
        # Fully connected layer: from 32 * 7 * 7 input features to 128 output features
        self.fc1 = nn.Linear(32 * 7 * 7, 128)  # Adjust input size based on your data
        # Fully connected layer: from 128 input features to 25 output features
        self.fc2 = nn.Linear(128, 25)

        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.batch_norm1(x)
        x = F.relu(x)
        x = self.pool1(x)

        x = self.conv2(x)
        x = self.batch_norm2(x)
        x = F.relu(x)
        x = self.pool2(x)

        x = x.view(x.size(0), -1)  # Flatten the tensor
        x = self.dropout(x)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        x = self.softmax(x)

        return x

    def test(self, predictions, labels):
        self.eval()
        correct = 0
        for p, l in zip(predictions, labels):
            if p == l:
                correct += 1
        acc = correct / len(predictions)
        return acc, correct, len(predictions)

    def evaluate(self, predictions, labels):
        correct = 0
        for p, l in zip(predictions, labels):
            if p == l:
                correct += 1
        acc = correct / len(predictions)
        return acc


def save_model(model, model_path):
    try:
        torch.save(model.state_dict(), model_path)
        torch.save(model, 'full_model-v2.pt')
        print("Model saved successfully")
    except Exception as e:
        print("Error saving model:", e)

def load_model(model_path):
    # try:
        # Instantiate the model
        model = Network()
        
        # Load the state dictionary from the given path
        model = torch.load("./"+model_path, map_location=torch.device("cuda" if torch.cuda.is_available() else "cpu"))

        model.eval()
        print("Model loaded successfully")
        return model


if __name__ == '__main__':
     model = load_model('model.pt')
     print(model.eval())