# import torch
# import torch.nn as nn
# import torch.nn.functional as F

# class Network(nn.Module):
#     def __init__(self):
#         super(Network, self).__init__()

#         # Define the layers of your neural network
#         self.conv1 = nn.Conv2d(1, 80, kernel_size=5)
#         self.conv2 = nn.Conv2d(80, 80, kernel_size=5)

#         self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
#         self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

#         self.batch_norm1 = nn.BatchNorm2d(80)
#         self.batch_norm2 = nn.BatchNorm2d(80)

#         self.fc1 = nn.Linear(1280, 250)
#         self.fc2 = nn.Linear(250, 25)

#     def forward(self, x):
#         # Define the forward pass of your neural network
#         x = self.conv1(x)
#         x = self.batch_norm1(x)
#         x = F.relu(x)
#         x = self.pool1(x)

#         x = self.conv2(x)
#         x = self.batch_norm2(x)
#         x = F.relu(x)
#         x = self.pool2(x)

#         x = x.view(x.size(0), -1)

#         x = F.relu(self.fc1(x))
#         x = self.fc2(x)
#         x = F.log_softmax(x, dim=1)

#         return x

# def load_model(model_path):
#     # Load the pre-trained model
#     try:
#         model = Network()  # Instantiate the model
#         model.load_state_dict(torch.load(model_path, map_location=torch.device("cuda" if torch.cuda.is_available() else "cpu")))
#         model.eval()  # Set the model to evaluation mode
#         print("Model loaded successfully")
#         return model
#     except Exception as e:
#         print("Error loading model:", e)
