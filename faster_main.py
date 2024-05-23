# <نسخة اسرع>
# <يوجد خطا في ثبات ترجمة الاشارات>
# <يوجد تاخر في نطق الترجمة مقارنة بسرعة معالجة الاشارات وطباعتها> [تمت معالجته]

import cv2
import mediapipe as mp
import torch
import numpy as np
import pyttsx3
from model import Network
import threading
from queue import Queue
import time

print("Packages imported...")

try:
    model_path = "AITP_Training_code.pt"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Network().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    print("Model loaded successfully")
except Exception as e:
    print("Error loading model:", e)

cam = cv2.VideoCapture(0)
cam.set(3, 320)  # خفض دقة العرض
cam.set(4, 240)  # خفض دقة الارتفاع

mpHands = mp.solutions.hands
hands = mpHands.Hands(max_num_hands=1)  # تقليل عدد الأيدي المعالجة
mpdraw = mp.solutions.drawing_utils

signs = {
    '0': 'A', '1': 'B', '2': 'C', '3': 'D', '4': 'E', '5': 'F', '6': 'G', '7': 'H', '8': 'I',
    '10': 'K', '11': 'L', '12': 'M', '13': 'N', '14': 'O', '15': 'P', '16': 'Q', '17': 'R',
    '18': 'S', '19': 'T', '20': 'U', '21': 'V', '22': 'W', '23': 'X', '24': 'Y'
}

def Transform(image):
    image = image / 255.0
    image = np.expand_dims(image, axis=0)
    image = np.expand_dims(image, axis=0)
    return torch.from_numpy(image).float().to(device)

frame_queue = Queue(maxsize=1)
output_queue = Queue(maxsize=1)
text_queue = Queue(maxsize=1)

engine = pyttsx3.init()
engine.setProperty('rate', 125)  # تقليل سرعة الكلام

def capture_frames():
    while True:
        ret, frame = cam.read()
        if not ret:
            break
        if frame_queue.empty():
            frame_queue.put(frame)
        if cv2.waitKey(1) & 0xff == 27:
            break

def process_frames():
    last_label = ""
    label_duration = 0.2  # مدة عرض كل ترجمة بالثواني
    last_time = time.time()

    while True:
        frame = frame_queue.get()
        if frame is None:
            break
        h, w, _ = frame.shape
        framergb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(framergb)
        hand_landmarks = result.multi_hand_landmarks

        if hand_landmarks:
            for handLMs in hand_landmarks:
                mpdraw.draw_landmarks(frame, handLMs, mpHands.HAND_CONNECTIONS)

            x_max, y_max = 0, 0
            x_min, y_min = w, h
            for handLMs in hand_landmarks:
                for lm in handLMs.landmark:
                    x, y = int(lm.x * w), int(lm.y * h)
                    x_max, y_max = max(x_max, x), max(y_max, y)
                    x_min, y_min = min(x_min, x), min(y_min, y)

            x_min, x_max = max(x_min - 10, 0), min(x_max + 10, w)
            y_min, y_max = max(y_min - 10, 0), min(y_max + 10, h)
            cropped_image = frame[y_min:y_max, x_min:x_max]

            try:
                image = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2GRAY)
                image = cv2.resize(image, (28, 28))
                image = Transform(image)
                with torch.no_grad():
                    prediction = model(image)
                prediction = torch.nn.functional.softmax(prediction, dim=1)
                i = prediction.argmax(dim=-1).cpu()
                label = signs[str(i.item())]
            except Exception as e:
                label = 'No Sign'

            if label != last_label and time.time() - last_time >= label_duration:
                last_label = label
                last_time = time.time()
                if text_queue.empty():
                    text_queue.put(label)

            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
            cv2.putText(frame, last_label, (x_min, y_min - 10), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 255, 0), 1)

        if output_queue.empty():
            output_queue.put(frame)

def speak_text():
    while True:
        text = text_queue.get()
        if text is None:
            break
        engine.say(text)
        engine.runAndWait()

capture_thread = threading.Thread(target=capture_frames)
process_thread = threading.Thread(target=process_frames)
speak_thread = threading.Thread(target=speak_text)

capture_thread.start()
process_thread.start()
speak_thread.start()

while True:
    frame = output_queue.get()
    if frame is None:
        break
    cv2.imshow("Output", frame)
    if cv2.waitKey(1) & 0xff == 27:
        break

frame_queue.put(None)
output_queue.put(None)
text_queue.put(None)

capture_thread.join()
process_thread.join()
speak_thread.join()

cam.release()
cv2.destroyAllWindows()
