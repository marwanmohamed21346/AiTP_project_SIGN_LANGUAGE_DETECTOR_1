import cv2
import mediapipe as mp
import torch
import pyttsx3
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import os

print("Packages imported...")

# Define the neural network class
class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()
        # Define the layers of your neural network
        self.conv = nn.Conv2d(1, 80, kernel_size=5)
        self.fc1 = nn.Linear(80 * 12 * 12, 250)
        self.fc2 = nn.Linear(250, 25)

    def forward(self, x):
        # Define the forward pass of your neural network
        x = self.conv(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = x.view(-1, 80 * 12 * 12)  # Flatten
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        x = F.log_softmax(x, dim=1)
        return x

# Save and load model path
model_path = "AITP_Training_code.pt"
model = Network()

# Check if the model exists, if not, train and save it
if os.path.exists(model_path):
    try:
        model.load_state_dict(torch.load(model_path, map_location=torch.device("cuda" if torch.cuda.is_available() else "cpu")))
        model.eval()
        print("Model loaded successfully")
    except Exception as e:
        print("Error loading model:", e)
else:
    # Placeholder training code
    # Normally you would have your own training loop here
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    dummy_data = torch.randn(10, 1, 28, 28)  # Dummy data for training
    dummy_target = torch.randint(0, 25, (10,))  # Dummy targets for training

    model.train()
    for epoch in range(1):
        optimizer.zero_grad()
        output = model(dummy_data)
        loss = criterion(output, dummy_target)
        loss.backward()
        optimizer.step()
    
    torch.save(model.state_dict(), model_path)
    print("Model trained and saved successfully")

# Initialize webcam
cam = cv2.VideoCapture(0)
cam.set(3, 700)  # Set width
cam.set(4, 480)  # Set height

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
mp_draw = mp.solutions.drawing_utils

# Initialize Pyttsx3
engine = pyttsx3.init()

# Dictionary for sign labels
signs = {
    '0': 'A', '1': 'B', '2': 'C', '3': 'D', '4': 'E', '5': 'F', '6': 'G', '7': 'H', '8': 'I',
    '10': 'K', '11': 'L', '12': 'M', '13': 'N', '14': 'O', '15': 'P', '16': 'Q', '17': 'R',
    '18': 'S', '19': 'T', '20': 'U', '21': 'V', '22': 'W', '23': 'X', '24': 'Y'
}

def transform(image):
    image = image / 255.0
    image = np.expand_dims(image, axis=0)
    image = np.expand_dims(image, axis=0)
    return torch.from_numpy(image).float()

# Main loop
while True:
    ret, frame = cam.read()
    if not ret:
        break

    h, w, _ = frame.shape
    framergb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(framergb)
    hand_landmarks = result.multi_hand_landmarks

    if hand_landmarks:
        for handLMs in hand_landmarks:
            mp_draw.draw_landmarks(frame, handLMs, mp_hands.HAND_CONNECTIONS)

        x_max, y_max = 0, 0
        x_min, y_min = w, h
        combined = [lm for handLMs in hand_landmarks for lm in handLMs.landmark]

        for lm in combined:
            x, y = int(lm.x * w), int(lm.y * h)
            x_max, y_max = max(x_max, x), max(y_max, y)
            x_min, y_min = min(x_min, x), min(y_min, y)

        x_min, x_max = max(x_min - 10, 0), min(x_max + 10, w)
        y_min, y_max = max(y_min - 10, 0), min(y_max + 10, h)
        cropped_image = frame[y_min:y_max, x_min:x_max]

        try:
            image = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2GRAY)
            image = cv2.resize(image, (28, 28))
            image = transform(image)
            with torch.no_grad():
                prediction = model(image)
            prediction = torch.nn.functional.softmax(prediction, dim=1)
            i = prediction.argmax(dim=-1).cpu()
            label = signs.get(str(i.item()), 'No Sign')
        except Exception as e:
            label = 'No Sign'

        cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
        cv2.putText(frame, label, (x_min, y_min - 10), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 255, 0), 1)
        
        if label != 'No Sign':
            engine.say(label)
            engine.runAndWait()

    cv2.imshow("Output", frame)
    if cv2.waitKey(1) & 0xFF == 27:  # Exit on ESC key
        break

cam.release()
cv2.destroyAllWindows()
