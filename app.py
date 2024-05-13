import cv2
import mediapipe as mp
import numpy as np
import torch
import os
from model import Network
print("Packages imported....")

try:
    model_path = "AITP_Training_code.pt"
    model = torch.load(model_path, map_location=torch.device("cuda"))
    model.eval()
    print("Model loaded successfully")
except Exception as e:
    print("Error loading model:", e)

cam = cv2.VideoCapture(0)
cam.set(3, 1920)  # Setting width
cam.set(4, 1080)  # Setting height

mpHands = mp.solutions.hands
hands = mpHands.Hands()
mpdraw = mp.solutions.drawing_utils

signs = {
    '0': 'A', '1': 'B', '2': 'C', '3': 'D', '4': 'E', '5': 'F', '6': 'G', '7': 'H', '8': 'I',
    '10': 'K', '11': 'L', '12': 'M', '13': 'N', '14': 'O', '15': 'P', '16': 'Q', '17': 'R',
    '18': 'S', '19': 'T', '20': 'U', '21': 'V', '22': 'W', '23': 'X', '24': 'Y'
}

while True:
    ret, frame = cam.read()
    if not ret:
        print("Failed to capture frame")
        break

    frame = cv2.flip(frame, 1)
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(img_rgb)

    inscreenlist = []
    if result.multi_hand_landmarks:
        for handInScreen in result.multi_hand_landmarks:
            for id, inscreen in enumerate(handInScreen.landmark):
                h, w, c = frame.shape
                cx, cy = int(inscreen.x * w), int(inscreen.y * h)
                inscreenlist.append([id, cx, cy])
                print(inscreenlist)
                mpdraw.draw_landmarks(frame, handInScreen, mpHands.HAND_CONNECTIONS)
        print("Hand detected")
        
        for label, sign in signs.items():
            font = cv2.FONT_HERSHEY_SIMPLEX
            frame = cv2.putText(frame, sign, (50, 50), font, 1, (255, 0, 0), 2, cv2.LINE_AA)

    cv2.imshow("Sign Language Detector", frame)
    if cv2.waitKey(1) & 0xff == 27:
        break

cam.release()
cv2.destroyAllWindows()
