try:
    import cv2
    import mediapipe as mp
    import torch
    import pyttsx3
    import numpy as np
    from model import Network
    from model import load_model
    print("Packages imported...")
except Exception as e:
    print("error loading packages:",e)

try:
    model_path = "Jo_elMofaker.pt"
    model = Network()
    print(model)
    model.eval()
    print("Model loaded successfully")
except Exception as e:
    print("Error loading model:", e)

cam = cv2.VideoCapture(0)
cam.set(3, 1280)  # Set width
cam.set(4, 720)  # Set height

mpHands = mp.solutions.hands
hands = mpHands.Hands()
mpdraw = mp.solutions.drawing_utils

signs = {
    '0': 'A',
    '1': 'B', 
    '2': 'C', 
    '3': 'D', 
    '4': 'E', 
    '5': 'F', 
    '6': 'G', 
    '7': 'H', 
    '8': 'I',
    '10': 'K', 
    '11': 'L', 
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
    '24': 'Y',
    ' ' : 'NO sign'
    }

def Transform(image):
    image = image / 255.0
    image = np.expand_dims(image, axis=0)
    image = np.expand_dims(image, axis=0)
    return torch.from_numpy(image).float()

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
            mpdraw.draw_landmarks(frame, handLMs, mpHands.HAND_CONNECTIONS)
            
            if len(hand_landmarks) == 1:
                x_max, y_max = 0, 0
                x_min, y_min = w, h
                combined = [lm for handLMs in hand_landmarks for lm in handLMs.landmark]

            for lm in combined:
                x, y = int(lm.x * w), int(lm.y * h)
                x_max, y_max = max(x_max, x), max(y_max, y)
                x_min, y_min = min(x_min, x), min(y_min, y)
        else:
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
            image = np.array(cv2.resize(image, (28, 28)))
            print(image.shape)
            image = Transform(image)
            with torch.no_grad():
                prediction = model(image)
            prediction = torch.nn.functional.softmax(prediction, dim=1)
            i = prediction.argmax(dim=-1).cuda()
            print(i)
            label = signs[str(i.item())]
        except Exception as e:
            label = 'No Sign'

        cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
        cv2.putText(frame, label, (x_min, y_min - 10), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 255, 0), 1)
        
        if label != 'No Sign':
            engine = pyttsx3.init()
            engine.say(label)
            engine.runAndWait()

    cv2.imshow("Output", frame)
    if cv2.waitKey(1) & 0xff == 27:
        break

cam.release()
cv2.destroyAllWindows()
