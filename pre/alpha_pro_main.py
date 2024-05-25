import time
import numpy as np
import cv2
import torch
import mediapipe as mp
import pyttsx3

try:
    from model import Net
    print("Package imported successfully...")
except Exception as e:
    print("Error loading packages!!!", e)

# Load the trained model
try:
    model = torch.load('model_trained.pt', map_location=torch.device('cpu'))  # Ensuring compatibility with CPU
    model.eval()
    print("Model loaded successfully...")
except Exception as e:
    print("Error loading model!!!", e)

# Define sign mappings
signs = {
    '0': 'A',
    '1': 'B',
    '2': 'welcome to',
    '3': 'D',
    '4': 'E',
    '5': 'F',
    '6': 'G',
    '7': 'H',
    '8': 'I',
    '10': 'K',
    '11': 'borg eLarab technology university',
    '12': 'M',
    '13': 'N',
    '14': 'O',
    '15': 'P',
    '16': 'Q',
    '17': 'R',
    '18': 'S',
    '19': 'T',
    '20': 'U',
    '21': 'V',
    '22': 'W',
    '23': 'X',
    '24': 'Y'
}

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Initialize text-to-speech engine
engine = pyttsx3.init()

# Initialize the webcam
cap = cv2.VideoCapture(0)
cap.set(3, 700)  # Set width
cap.set(4, 480)  # Set height

predicted_sign = "Sign not detected"
last_prediction_time = time.time()
prediction_interval = 0.2  # Time interval for predictions in seconds

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to capture image from webcam. Exiting...")
        break

    # Convert frame to RGB for processing
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the frame with MediaPipe Hands
    results = hands.process(rgb_frame)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Calculate bounding box around hand landmarks
            x_min, y_min = int(min([lm.x for lm in hand_landmarks.landmark]) * frame.shape[1]), int(min([lm.y for lm in hand_landmarks.landmark]) * frame.shape[0])
            x_max, y_max = int(max([lm.x for lm in hand_landmarks.landmark]) * frame.shape[1]), int(max([lm.y for lm in hand_landmarks.landmark]) * frame.shape[0])

            # Ensure bounding box is within frame dimensions
            x_min, y_min = max(0, x_min), max(0, y_min)
            x_max, y_max = min(frame.shape[1], x_max), min(frame.shape[0], y_max)

            # Extract the hand region
            hand_img = frame[y_min:y_max, x_min:x_max]

            # Preprocess the hand image for model prediction
            res = cv2.resize(hand_img, (28, 28))
            res = cv2.cvtColor(res, cv2.COLOR_BGR2GRAY)
            res = np.reshape(res, (1, 1, 28, 28)) / 255.0
            res = torch.from_numpy(res).float()

            # Perform model prediction at defined intervals
            current_time = time.time()
            if current_time - last_prediction_time >= prediction_interval:
                last_prediction_time = current_time

                with torch.no_grad():
                    out = model(res)
                    probs = torch.nn.functional.softmax(out, dim=1)
                    pred = out.argmax(dim=1, keepdim=True)

                if probs[0, pred] < 0.4:
                    predicted_sign = 'Sign not detected'
                else:
                    predicted_sign = signs.get(str(pred.item()), 'Unknown')

            # Display the prediction on the frame
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(frame, predicted_sign, (60, 60), font, 1, (255, 0, 0), 2, cv2.LINE_AA)

    # Display the frame with hand landmarks and prediction
    cv2.imshow('Sign Language Recognition', frame)

    # Handle key events
    key = cv2.waitKey(1)
    if key & 0xFF == 27:  # ESC key to exit
        break
    elif key & 0xFF == ord(' '):  # Spacebar to speak the predicted sign
        engine.say(predicted_sign)
        engine.runAndWait()

# Release resources
cap.release()
cv2.destroyAllWindows()